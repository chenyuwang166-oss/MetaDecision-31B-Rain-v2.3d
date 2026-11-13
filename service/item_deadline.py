"""Per-decision deadline for the drafting pass, plus two item rules (installed by server.py when
config.json has an `item_deadline` block, with the `item_rules` block; no package file is modified at
runtime).

One hard deadline per request (the harness sends one decision per request): the moment the service
starts answering the request + item_seconds. The draft is generated greedily in chunks of at most
`chunk_tokens`; each chunk is sized by a conservative speed model (a short synthetic warm-up at
startup, updated online) so that it ends before deadline - response reserve - readout reserve.
No engine call is abandoned in flight, so every call is metered exactly, and the model lock is
released when the decision returns.
- Draft budget used up without the answer marker: the marker is appended, exactly as when
  max_new_tokens is reached.
- Readout that cannot finish before the deadline: the first-pass answer is returned (route
  slow_timeout_fast) and the decision's usage is reconciled exactly from its meters.
- An answer marker that straddles a chunk boundary ends the draft at the token that completes it,
  where one-shot generation would have stopped.
Decisions that do not take the drafting pass are unaffected by the deadline.

Item rules (wrap_service):
- slow_types: question types that may take the drafting pass (None = all). For any other type the
  routing threshold is 0 for that decision: it never requests the drafting pass and the quota
  counter never counts it.
- reread_tau: a choice decision answered in one pass although its length allowed the drafting pass
  (budget tier not 'long', not a long item), whose first-pass margin is below tau, is read once
  more, with the options in reverse order (each label keeps its criterion); the two answer
  distributions are averaged by label. The second read is metered into the
  decision's usage (so the cost fuse sees it) and is not a quota decision. If it fails (not an
  engine failure) the first-pass answer is kept and `reread_failed` is recorded.
Controller.truncated_readout is a diagnostic used only by offline measurement; serving never calls it.
"""
import math
import re
import threading
import time
from contextlib import contextmanager

DEFAULTS = dict(item_seconds=9.0, response_reserve_s=0.35, readout_reserve_init_s=1.0,
                readout_reserve_min_s=0.3, readout_safety=1.5, chunk_tokens=32, min_chunk_tokens=8,
                token_safety=1.25, ewma_alpha=0.3, warmup_tokens=33, warmup_repeats=3)

_local = threading.local()


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def settings_from(values=None):
    s = dict(DEFAULTS)
    s.update(values or {})
    need(set(s) == set(DEFAULTS), 'DEADLINE_SETTINGS_KEYS')
    for key in ('item_seconds', 'response_reserve_s', 'readout_reserve_init_s', 'readout_reserve_min_s',
                'readout_safety', 'token_safety', 'ewma_alpha'):
        x = s[key]
        need(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0,
             'DEADLINE_SETTING_' + key.upper())
    for key in ('chunk_tokens', 'min_chunk_tokens', 'warmup_tokens', 'warmup_repeats'):
        need(type(s[key]) is int and s[key] > 0, 'DEADLINE_SETTING_' + key.upper())
    need(s['token_safety'] >= 1 and s['readout_safety'] >= 1 and s['ewma_alpha'] <= 1, 'DEADLINE_SAFETY')
    need(s['min_chunk_tokens'] <= s['chunk_tokens'] and s['warmup_tokens'] >= 2 and s['warmup_repeats'] >= 2,
         'DEADLINE_TOKEN_LIMITS')
    need(s['item_seconds'] > s['response_reserve_s'] + s['readout_reserve_init_s'], 'DEADLINE_RESERVES')
    return s


class SpeedModel:
    """Conservative chunk planner: seconds(n) = call_s + n * token_s * token_safety."""

    def __init__(self, token_s, call_s, settings):
        need(all(isinstance(x, float) and math.isfinite(x) and x > 0 for x in (token_s, call_s)),
             'DEADLINE_SPEED')
        self.s = settings
        self.lock = threading.Lock()
        self.warm_token_s = token_s
        self.token_s = token_s
        self.call_s = call_s
        self.ewma = None
        self.readout = {}
        self.chunks_observed = 0

    def fit_tokens(self, seconds):
        if not math.isfinite(seconds) or seconds <= self.call_s:
            return 0
        return int(math.floor((seconds - self.call_s) / (self.token_s * self.s['token_safety'])))

    def observe_chunk(self, generated, seconds):
        if generated < 4 or not math.isfinite(seconds) or seconds <= 0:
            return
        per = max(0., seconds - self.call_s) / generated
        with self.lock:
            a = self.s['ewma_alpha']
            self.ewma = per if self.ewma is None else a * per + (1 - a) * self.ewma
            self.token_s = max(self.warm_token_s, self.ewma)
            self.chunks_observed += 1

    def readout_reserve(self, kind):
        v = self.readout.get(kind)
        if v is None:
            return self.s['readout_reserve_init_s']
        return max(self.s['readout_reserve_min_s'], self.s['readout_safety'] * v)

    def observe_readout(self, kind, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self.lock:
            a = self.s['ewma_alpha']
            old = self.readout.get(kind)
            self.readout[kind] = seconds if old is None else a * seconds + (1 - a) * old

    def snapshot(self):
        return dict(token_s=self.token_s, warm_token_s=self.warm_token_s, call_s=self.call_s,
                    chunks_observed=self.chunks_observed, readout={k: v for k, v in sorted(self.readout.items())})


class ItemPlan:
    def __init__(self, controller, start, max_new_tokens, kind):
        s = controller.settings
        self.c = controller
        self.start = start
        self.kind = kind
        self.max_new_tokens = max_new_tokens
        self.hard = start + s['item_seconds']
        self.meter_deadline = self.hard - s['response_reserve_s']
        self.meters = []
        self.chunks = []
        self.budget_stop = False
        self.straddle = False
        self.forced = False
        self.draft_tokens = 0
        self.slow_started = None
        self.draft_end_at = None
        self.readout_start = None
        self.readout_s = None
        self.timeout_phase = None
        self.reconciled = False
        # draft bookkeeping (memory only; never serialised)
        self._prompt_ids = None
        self._draft_ids = None
        self._by_marker = False
        self._groups = None
        self._config = None
        self._slow_meter = None

    def draft_deadline(self):
        return self.meter_deadline - self.c.speed.readout_reserve(self.kind)

    def exact(self):
        return all(not m.get('engine_usage_unknown') and not m.get('cached_unknown') for m in self.meters)

    def audit(self, now):
        def rel(t):
            return None if t is None else round(t - self.start, 6)
        speed = self.c.speed
        return dict(schema='item-deadline-v1', item_seconds=self.c.settings['item_seconds'], elapsed_s=rel(now),
                    slow_started_s=rel(self.slow_started), draft_end_s=rel(self.draft_end_at),
                    readout_s=None if self.readout_s is None else round(self.readout_s, 6),
                    draft_tokens=self.draft_tokens, chunks=len(self.chunks),
                    chunk_tokens=[c['generated'] for c in self.chunks][:64],
                    chunk_seconds=[round(c['seconds'], 6) for c in self.chunks][:64],
                    chunk_end_s=[round(c['end'] - self.start, 6) for c in self.chunks][:64],
                    chunk_cum_tokens=[c['cum_tokens'] for c in self.chunks][:64],
                    chunk_cum_input=[c['cum_input'] for c in self.chunks][:64],
                    chunk_cum_generated=[c['cum_generated'] for c in self.chunks][:64],
                    draft_by_marker=bool(self._by_marker),
                    budget_stop=self.budget_stop, straddle=self.straddle, forced=self.forced,
                    timeout_phase=self.timeout_phase, reconciled=self.reconciled, exact=self.exact(),
                    max_new_tokens=self.max_new_tokens, token_s=round(speed.token_s, 6),
                    call_s=round(speed.call_s, 6), readout_reserve_s=round(speed.readout_reserve(self.kind), 6))


def reconcile(detail, meters):
    """Usage arithmetic of core.py _one_impl, without the timeout estimate."""
    submitted = sum(m['engine_submitted'] for m in meters)
    cached = sum(m['cached'] for m in meters)
    generated = sum(m.get('engine_generated', m['output']) for m in meters)
    engine_input = sum(m.get('engine_input', m['engine_submitted'] - m['cached']) for m in meters)
    detail['usage_estimated'] = False
    detail['input_tokens_details'].update(
        cached_tokens=cached, engine_submitted_tokens=submitted, engine_generated_tokens=generated,
        engine_calls=sum(m.get('engine_calls', 0) for m in meters), cached_tokens_known=True)
    detail['usage_bases'] = {
        'engine': dict(input_tokens=engine_input, output_tokens=generated),
        'each_once': dict(input_tokens=detail['prompt_tokens'], output_tokens=detail['completion_tokens']),
        'submitted': dict(input_tokens=submitted, output_tokens=detail['completion_tokens'])}


def marker_completion(decode, previous, new, marker):
    """Smallest j such that the marker is complete in decode(previous+new[:j]); None if absent."""
    if marker not in decode(list(previous) + list(new)):
        return None
    for j in range(1, len(new) + 1):
        if marker in decode(list(previous) + list(new[:j])):
            return j
    return None


@contextmanager
def request_clock(start=None):
    previous = getattr(_local, 'request_start', None)
    _local.request_start = time.monotonic() if start is None else start
    try:
        yield
    finally:
        _local.request_start = previous


def active_plan():
    return getattr(_local, 'plan', None)


def _install_core(core):
    if getattr(core, '_deadline_core', None):
        return
    original_bounded = core.bounded_slow_attempt
    original_new_meter = core.new_meter

    def bounded_slow_attempt(backend, ids, groups, config, meter, model_lock, *, on_service_failure=None):
        plan = getattr(_local, 'plan', None)
        if plan is None:
            return original_bounded(backend, ids, groups, config, meter, model_lock,
                                    on_service_failure=on_service_failure)
        return plan.c.bounded(plan, backend, ids, groups, config, meter, model_lock, on_service_failure)

    def new_meter():
        m = original_new_meter()
        plan = getattr(_local, 'plan', None)
        if plan is not None:
            plan.meters.append(m)
        return m

    core.bounded_slow_attempt = bounded_slow_attempt
    core.new_meter = new_meter
    core._deadline_core = dict(bounded_slow_attempt=original_bounded, new_meter=original_new_meter)


def _install_backend(backend):
    if getattr(backend, '_deadline_backend', None):
        return
    original = backend.slow_session

    def slow_session(ids, unparsed_policy, meter, max_new_tokens=None):
        plan = getattr(_local, 'plan', None)
        if plan is None:
            return original(ids, unparsed_policy, meter, max_new_tokens=max_new_tokens)
        need(plan.c.backend is backend, 'DEADLINE_BACKEND_MISMATCH')
        return plan.c.chunked_session(plan, ids, unparsed_policy, meter, max_new_tokens)

    backend.slow_session = slow_session
    backend._deadline_backend = dict(slow_session=original)


def measure_speed(backend, settings, system_prompt, clock=time.monotonic):
    """Synthetic prompt only. Paired 1/N exact-length greedy calls; conservative maxima."""
    g = backend.generation
    ids = backend.prompt_ids(system_prompt, 'Ready?', False)
    n_tokens = settings['warmup_tokens']
    pairs = []

    def run(n):
        params = backend.SamplingParams(max_tokens=n, min_tokens=n, temperature=0.0, top_p=1.0, top_k=-1,
                                        repetition_penalty=g['repetition_penalty'], seed=g['seed'],
                                        detokenize=False)
        t0 = clock()
        out = backend._completion(ids, params, None, 'draft')
        dt = clock() - t0
        need(len(out.token_ids) == n, 'DEADLINE_WARMUP_LENGTH')
        return dt

    for _ in range(settings['warmup_repeats']):
        one = run(1)
        many = run(n_tokens)
        pairs.append((one, many))
    call_s = float(max(x for x, _ in pairs))
    token_s = float(max(max((y - x) / (n_tokens - 1), y / n_tokens) for x, y in pairs))
    token_s = max(token_s, 1e-6)
    call_s = max(call_s, 1e-6)
    return SpeedModel(token_s, call_s, settings), dict(pairs_s=pairs, tokens=n_tokens)


class Controller:
    def __init__(self, core, backends, backend, settings, speed, clock=time.monotonic, warmup=None):
        self.core, self.backends, self.backend = core, backends, backend
        self.settings, self.speed, self.clock = settings, speed, clock
        self.warmup = warmup

    # ---- per item ---------------------------------------------------------------------------
    def slow_attempt(self, backend, ids, groups, config, meter):
        """core.slow_attempt semantics (same exception mapping), except that the deadline
        path skips release_failed_session's gc.collect(): on vLLM there is no allocator state to
        clear, and a full collection would only delay the fast answer past the item cap."""
        core = self.core
        failed = None
        try:
            return core.slow_readout(backend, ids, groups, config, meter)
        except TimeoutError:
            failed = 'timeout'
        except Exception as exc:
            if core.unrecoverable_engine_error(exc):
                raise core.ServiceFailure('unrecoverable engine failure') from exc
            failed = 'error'
        if failed == 'timeout':
            raise TimeoutError('Slow deadline exceeded')
        core.release_failed_session(backend)
        raise RuntimeError('Slow session failed')

    def bounded(self, plan, backend, ids, groups, config, meter, model_lock, on_service_failure):
        core = self.core
        plan._groups, plan._config, plan._slow_meter = groups, config, meter
        meter['deadline'] = plan.meter_deadline
        plan.slow_started = self.clock()
        try:
            with model_lock:
                core.check_deadline(meter)
                value = self.slow_attempt(backend, ids, groups, config, meter)
                core.check_deadline(meter)
            if plan.readout_start is not None:
                plan.readout_s = self.clock() - plan.readout_start
                self.speed.observe_readout(plan.kind, plan.readout_s)
            return value
        except Exception as exc:
            if core.unrecoverable_engine_error(exc) and on_service_failure is not None:
                on_service_failure()
            if isinstance(exc, TimeoutError):
                plan.timeout_phase = 'readout' if plan.readout_start is not None else 'draft'
            raise

    def chunked_session(self, plan, ids, unparsed_policy, meter, max_new_tokens=None):
        backend, s, speed = self.backend, self.settings, self.speed
        g = backend.generation
        need(g['do_sample'] is False and g['repetition_penalty'] == 1.0, 'DEADLINE_CHUNKS_REQUIRE_GREEDY')
        count = self.backends._max_new_tokens(max_new_tokens or plan.max_new_tokens)
        marker = backend.marker
        prompt = list(ids)
        gen = []
        by_marker = False
        counted_prompt = False
        while len(gen) < count:
            now = self.clock()
            if now >= plan.meter_deadline:
                raise TimeoutError('item deadline before chunk')
            left = count - len(gen)
            n = min(s['chunk_tokens'], left, speed.fit_tokens(plan.draft_deadline() - now))
            if n < min(s['min_chunk_tokens'], left):
                if not gen:
                    raise TimeoutError('no draft budget')
                plan.budget_stop = True
                break
            params = backend.SamplingParams(max_tokens=n, temperature=0.0, top_p=1.0, top_k=-1,
                                            repetition_penalty=g['repetition_penalty'], seed=g['seed'],
                                            stop=[marker], include_stop_str_in_output=True)
            if not counted_prompt:
                meter['prompt'] += len(prompt)   # the backend's slow_session counts the prompt once
                counted_prompt = True
            t0 = self.clock()
            out = backend._completion(prompt + gen, params, meter, 'draft')
            t1 = self.clock()
            dt = t1 - t0
            new = list(out.token_ids)
            speed.observe_chunk(len(new), dt)
            plan.chunks.append(dict(n=n, generated=len(new), seconds=dt, end=t1, cum_tokens=len(gen) + len(new),
                                    cum_input=int(meter.get('engine_input', 0)),
                                    cum_generated=int(meter.get('engine_generated', 0))))
            if getattr(out, 'stop_reason', None) == marker and getattr(out, 'finish_reason', None) == 'stop':
                gen += new
                by_marker = True
                break
            j = marker_completion(backend.decode, gen, new, marker)
            if j is not None:
                # One-shot generation would have stopped at the token completing the marker. The
                # extra tokens of this chunk were generated and stay metered (exact cost).
                gen += new[:j]
                by_marker = True
                plan.straddle = True
                break
            gen += new
            if getattr(out, 'finish_reason', None) == 'stop' or len(new) < n:
                break                        # EOS: the draft ended without the marker
        plan.draft_tokens = len(gen)
        plan.draft_end_at = self.clock()
        plan._prompt_ids, plan._draft_ids, plan._by_marker = list(prompt), list(gen), by_marker
        parsed = by_marker and bool(re.search(r'(?m)^[ \t]*' + re.escape(marker) + r'$', backend.decode(gen)))
        if not parsed:
            if unparsed_policy == 'fast':
                return None, False
            forced = backend.encode('\n' + marker)
            gen += forced
            meter['output'] += len(forced)
            meter['forced'] += len(forced)
            plan.forced = True
        plan.readout_start = self.clock()
        return self.backends.VLLMPrefixSession(backend, prompt + gen, meter), parsed

    # ---- truncated-draft readout (diagnostic for offline measurement; not used when serving) ----
    def truncated_readout(self, plan, n_tokens, labels, qtype, calibration_block):
        """Readout of this item's draft cut at n_tokens (no marker within them) + the forced answer
        marker, on a fresh meter: what a one-shot cap of n_tokens would have read out (greedy drafts
        are prefix-identical). Returns (probabilities dict, meter, seconds). Caller checks eligibility."""
        need(plan._draft_ids is not None and len(plan._draft_ids) > n_tokens, 'DEADLINE_TRUNC_NOT_LONGER')
        backend, core = self.backend, self.core
        forced = backend.encode('\n' + backend.marker)
        meter = self.backends.new_meter()
        meter['output'] += len(forced)
        meter['forced'] += len(forced)
        t0 = self.clock()
        session = self.backends.VLLMPrefixSession(backend, plan._prompt_ids + plan._draft_ids[:n_tokens] + forced, meter)
        scores, _ = core.sequence_scores(session, backend, plan._groups, meter)
        probs = core.probabilities(labels, scores, qtype, calibration_block)
        return probs, meter, self.clock() - t0

    # ---- low-margin choice re-read ------------------------------------------------------------
    def reread_reading(self, service, task):
        """One fast read of a choice task with the option order reversed (label -> criterion pairs kept;
        render re-letters the options), the service's 'fast' calibration. Returns (probabilities by label, meter,
        seconds). The meter is created through core.new_meter, so inside an item it joins the item's meters."""
        core = self.core
        need(task['question']['type'] == 'choice' and task.get('labels'), 'DEADLINE_REREAD_TASK')
        t2 = dict(task, labels=list(reversed(task['labels'])))
        meter = core.new_meter()
        t0 = self.clock()
        _, (labels2, groups2, fast_ids2, _), _ = core.prepare_task(t2, self.backend, service.config)
        with service.model_lock:
            scores2, _, _ = core.fast_attempt_with_retry(self.backend, t2, service.config, fast_ids2, labels2, groups2, meter)
        need(scores2 is not None, 'DEADLINE_REREAD_SCORES')
        probs = core.probabilities(labels2, scores2, 'choice', service.calibration['fast'])
        return probs, meter, self.clock() - t0

    @staticmethod
    def reread_due(task, detail, tau):
        m = detail.get('fast_margin')
        return (tau is not None and tau > 0 and task['question']['type'] == 'choice' and detail.get('path') == 'fast'
                and detail.get('budget_tier') != 'long' and not detail.get('long_item')
                and isinstance(m, (int, float)) and math.isfinite(m) and m < tau)

    def apply_reread(self, service, task, answer_value, detail):
        """Online re-read: average the fast answer with the reversed-order read; add the read to the item's
        usage (every basis) and token details; record it in detail['reread']."""
        probs, meter, seconds = self.reread_reading(service, task)
        first = answer_value['probabilities']
        need(set(first) == set(probs), 'DEADLINE_REREAD_LABELS')
        mixed = {k: (first[k] + probs[k]) / 2 for k in first}
        winner = max(task['labels'], key=mixed.get)
        engine_input = meter.get('engine_input', meter['engine_submitted'] - meter['cached'])
        generated = meter.get('engine_generated', meter['output'])
        bases = detail['usage_bases']
        bases['engine']['input_tokens'] += engine_input
        bases['engine']['output_tokens'] += generated
        bases['each_once']['input_tokens'] += meter['prompt']
        bases['each_once']['output_tokens'] += meter['output']
        bases['submitted']['input_tokens'] += meter['engine_submitted']
        bases['submitted']['output_tokens'] += meter['output']
        d = detail['input_tokens_details']
        d['cached_tokens'] += meter['cached']
        d['engine_submitted_tokens'] += meter['engine_submitted']
        d['engine_generated_tokens'] += generated
        d['engine_calls'] += meter.get('engine_calls', 0)
        if meter.get('cached_unknown'):
            d['cached_tokens_known'] = False
        if meter.get('engine_usage_unknown') or meter.get('cached_unknown'):
            detail['usage_estimated'] = True
        detail['prompt_tokens'] += meter['prompt']
        detail['completion_tokens'] += meter['output']
        detail['total_tokens'] = detail['prompt_tokens'] + detail['completion_tokens']
        detail['readout_output_tokens'] += meter['probes']
        detail['prediction'] = winner
        detail['reread'] = dict(input_tokens=int(engine_input), output_tokens=int(generated), seconds=seconds,
                                     order='reversed')
        answer_value = dict(answer_value, probabilities=mixed)
        answer_value['choice'] = winner
        return answer_value, detail

    # ---- per service ------------------------------------------------------------------------
    def wrap_service(self, service, slow_types=None, reread_tau=None):
        """slow_types: question types allowed on the slow path (None = all); reread_tau: low-margin choice
        re-read threshold (None or 0 = off)."""
        need(service.backend is self.backend, 'DEADLINE_SERVICE_BACKEND')
        need(not getattr(service, '_deadline_wrapped', False), 'DEADLINE_ALREADY_WRAPPED')
        need(slow_types is None or (set(slow_types) <= {'choice', 'noul', 'score'} and len(slow_types) > 0), 'DEADLINE_SLOW_TYPES')
        need(reread_tau is None or (isinstance(reread_tau, (int, float)) and math.isfinite(reread_tau) and reread_tau >= 0),
             'DEADLINE_REREAD_TAU')
        original_one, original_answer = service._one_impl, service.answer
        clock = self.clock
        allowed = None if slow_types is None else frozenset(slow_types)

        def answer(payload):
            owned = getattr(_local, 'request_start', None) is None
            if owned:
                _local.request_start = clock()
            try:
                return original_answer(payload)
            finally:
                if owned:
                    _local.request_start = None

        def one(task):
            need(getattr(_local, 'plan', None) is None, 'DEADLINE_NESTED_ITEM')
            start = getattr(_local, 'request_start', None)
            plan = ItemPlan(self, clock() if start is None else start,
                            service.config['generation']['max_new_tokens'], task['question']['type'])
            service._deadline_last = plan
            _local.plan = plan
            routing = service.config['routing']
            switched_off = allowed is not None and task['question']['type'] not in allowed
            saved = routing['margin_threshold']
            try:
                if switched_off:
                    routing['margin_threshold'] = 0.0      # margins are >= 0: never requests slow, never counted
                try:
                    answer_value, detail = original_one(task)
                finally:
                    routing['margin_threshold'] = saved
                if detail.get('route') == 'slow_timeout_fast' and plan.exact():
                    reconcile(detail, plan.meters)
                    plan.reconciled = True
                if self.reread_due(task, detail, reread_tau):
                    try:
                        answer_value, detail = self.apply_reread(service, task, answer_value, detail)
                    except Exception as exc:
                        if self.core.unrecoverable_engine_error(exc):
                            raise self.core.ServiceFailure('unrecoverable engine failure') from exc
                        detail['reread_failed'] = type(exc).__name__     # fast answer kept, as without re-read
                detail['item_deadline'] = plan.audit(clock())
                detail['item_deadline']['slow_type_off'] = switched_off
                return answer_value, detail
            finally:
                _local.plan = None

        service._one_impl = one
        service.answer = answer
        service._deadline_wrapped = True
        service._deadline_slow_types = None if allowed is None else sorted(allowed)
        service._deadline_reread_tau = reread_tau
        return service


def install(core, backends, backend, settings=None, system_prompt=None, clock=time.monotonic, speed=None):
    """Process-wide, idempotent dispatchers + one controller for this backend."""
    s = settings_from(settings)
    existing = getattr(backend, '_deadline_controller', None)
    if existing is not None:
        need(existing.settings == s, 'DEADLINE_SETTINGS_CHANGED')
        return existing
    _install_core(core)
    warm = None
    if speed is None:
        need(isinstance(system_prompt, str), 'DEADLINE_SYSTEM_PROMPT')
        speed, warm = measure_speed(backend, s, system_prompt, clock)
    _install_backend(backend)
    controller = Controller(core, backends, backend, s, speed, clock, warm)
    backend._deadline_controller = controller
    return controller
