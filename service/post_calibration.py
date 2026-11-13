"""Per-type post-calibration of the reported probabilities (no model call, no tokens).

server.py installs it when config.json has a `post_calibration` block. For each question type the
block holds the options and their parameters: chosen on the service's cached outputs and labels of
public and own development data, and partly set from or re-weighted to published aggregate statistics
of the official evaluation (no non-public evaluation question, gold answer or per-item result;
DISCLOSURE.md):
  choice: debias (divide out the mean probability of each option position, per option count, then
          renormalise), then a temperature, then optionally a fixed top mass;
  noul:   a temperature, then a decision-side floor (P(yes) is moved to at least `floor` on the
          side it already favours, or to at most 1 - floor);
  score:  a temperature, then a reading weight w (probability mass w moved onto the most probable
          level; w = 1 puts all mass on it); or, instead of both, a symmetric residual: the most
          probable level gets the fixed mass `top`, the rest goes to its adjacent levels (split
          equally between the two neighbours of an inner level, so the expected level is that level;
          a first or last level has one neighbour and is handled by the chosen endpoint rule:
          "uniform" gives it `top` too, "endpoint_onehot" keeps all mass on it, "endpoint_half"
          gives it 1 - (1 - top)/2).
The temperature option is off, one value per type ("type"), or one value per type and route cell
("route": fast, slow, slow_timeout_fast, fast_reread; a cell without its own value uses the type
value). A temperature t replaces the probabilities p by p^(1/t), renormalised. The temperature and
the symmetric residual check the distribution they are given (finite, 0..1, sum 1) before using it,
and every calibrated result is checked the same way; through the service hook a refused
distribution becomes an ItemFailure. A type without an
entry, or with every option off, and a question with a single option or level, are returned exactly
as the service computed them. The answer label (`choice`, `noul`, the prediction) is recomputed from
the calibrated probabilities.
"""
import math

KINDS = ('choice', 'noul', 'score')
OPTIONS = {'choice': ('debias', 'temperature', 'fixed_top'), 'noul': ('temperature', 'floor'),
           'score': ('temperature', 'mode_weight', 'symmetric_residual')}
SYMMETRIC_RESIDUAL_MODES = ('uniform', 'endpoint_onehot', 'endpoint_half')
ROUTE_BUCKETS = ('fast', 'slow', 'slow_timeout_fast', 'fast_reread')
TEMPERATURE_MODES = (False, True, 'type', 'route')
T_MIN, T_MAX = 0.5, 8.0


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- transforms
def distribution(p):
    need(len(p) >= 2 and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in p), 'BAD_PROBS')
    need(abs(sum(p)-1) < 1e-6, 'BAD_PROB_SUM')
    return list(p)


def softmax(scores):
    need(scores and all(math.isfinite(x) for x in scores), 'BAD_LOGITS')
    z = max(scores)
    w = [math.exp(x-z) for x in scores]
    return [x/sum(w) for x in w]


def temperature(p, t):
    distribution(p)
    need(math.isfinite(t) and t >= .5, 'BAD_TEMPERATURE')
    return list(p) if t == 1 else softmax([math.log(max(x, 1e-300))/t for x in p])


def _argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def choice_fixed_top(p, top):
    """Keep the first maximum; preserve non-winner ratios, or split a one-hot residual equally."""
    need(finite(top) and .5 < top < 1, 'CAL_CHOICE_TOP')
    if len(p) == 1:
        need(finite(p[0]) and p[0] == 1, 'BAD_PROBS')
        return list(p)
    p = distribution(p)
    a = _argmax(p)
    rest = sum(x for i, x in enumerate(p) if i != a)
    return distribution([top if i == a else (1 - top) * (x / rest if rest else 1 / (len(p) - 1))
                         for i, x in enumerate(p)])


def debias(p, prior):
    """Option-position prior divided out, renormalised."""
    need(len(prior) == len(p) and all(w > 0 for w in prior), 'CAL_PRIOR')
    q = [x / w for x, w in zip(p, prior)]
    s = sum(q)
    return [x / s for x in q]


def noul_floor(p, floor):
    """Labels no/yes: P(yes) is committed to >= floor on the favoured side (<= 1 - floor on the other)."""
    need(len(p) == 2 and .8 < floor < 1, 'CAL_NOUL_FLOOR')
    if p[1] >= floor or p[1] <= 1 - floor:
        return list(p)
    y = floor if p[1] >= .5 else 1 - floor
    return [1 - y, y]


def score_mode(p, weight):
    """Weight 0 keeps the distribution; weight 1 puts all mass on the most probable level."""
    need(0 <= weight <= 1, 'CAL_SCORE_WEIGHT')
    if weight == 0:
        return list(p)
    m = _argmax(p)
    return [(1 - weight) * x + (weight if i == m else 0.) for i, x in enumerate(p)]


def score_symmetric_residual(p, mode, top):
    """Levels in their order; the most probable level a keeps mass `top` (0.5 < top < 1), the residual
    1 - top goes to the adjacent levels: half to each neighbour of an inner level (expected level = a);
    a first/last level (one neighbour) follows `mode`. The most probable level never changes."""
    need(mode in SYMMETRIC_RESIDUAL_MODES and finite(top) and .5 < top < 1, 'CAL_SCORE_RESIDUAL')
    p = distribution(p)                                  # before the argmax: NaN/Inf/negative/sum != 1 refused
    k = len(p)
    a = _argmax(p)
    q = [0.] * k
    if 0 < a < k - 1:
        q[a] = top
        q[a - 1] = q[a + 1] = (1 - top) / 2
        return q
    keep = {'uniform': top, 'endpoint_onehot': 1., 'endpoint_half': 1 - (1 - top) / 2}[mode]
    q[a] = keep
    if keep < 1:
        q[1 if a == 0 else k - 2] = 1 - keep
    return q


def n_active(cfg):
    return sum(bool(v) for v in cfg.values())


def apply_one(kind, cfg, params, p, bucket='fast'):
    q = list(p)
    if kind == 'choice' and cfg.get('fixed_top') and len(q) == 1:
        return choice_fixed_top(q, params['top'])
    if cfg.get('debias') and str(len(q)) in params.get('prior', {}):
        q = debias(q, params['prior'][str(len(q))])
    mode = cfg.get('temperature')
    if mode:
        need(mode in (True, 'type', 'route'), 'CAL_TEMPERATURE_MODE')
        t = params['route_T'].get(bucket, params['T']) if mode == 'route' else params['T']
        q = temperature(q, t)
    if kind == 'noul' and cfg.get('floor'):
        q = noul_floor(q, cfg['floor'])
    if kind == 'score' and cfg.get('mode_weight'):
        q = score_mode(q, cfg['mode_weight'])
    if kind == 'score' and cfg.get('symmetric_residual'):
        q = score_symmetric_residual(q, cfg['symmetric_residual'], params['top'])
    if kind == 'choice' and cfg.get('fixed_top'):
        q = choice_fixed_top(p, params['top'])  # Last: the previous mapping overrides the route temperature.
    return distribution(q)


def route_bucket(detail):
    """Route cell of one answered decision, from the service's own per-decision record:
    slow_timeout_fast = first-pass answer served after an admitted drafting pass (deadline, failure or
    unparsed draft; the 'fallback' calibration block); slow = answer read after the drafting pass
    (appended answer line included); fast_reread = one-pass answer with the reversed-order re-read
    averaged in; fast = everything else."""
    if detail.get('route') in ('slow_timeout_fast', 'slow_failed_fast', 'slow_unparsed_fast') \
            or detail.get('calibration_block') == 'fallback':
        return 'slow_timeout_fast'
    if detail.get('path') == 'slow' or detail.get('calibration_block') == 'slow' \
            or detail.get('route') in ('slow', 'slow_forced_answer'):
        return 'slow'
    if isinstance(detail.get('reread'), dict):
        return 'fast_reread'
    return 'fast'


# ---------------------------------------------------------------- configuration
def _temperature_value(t):
    return finite(t) and T_MIN <= t <= T_MAX


def validate(block):
    """The config.json block: {type: {"config": {option: value},
    "params": {"T": float, "route_T": {cell: float}, "route_counts": {cell: int}, "prior": {...},
    "top": float}}}; "top" is the most-probable-level mass of the score symmetric residual."""
    need(isinstance(block, dict) and set(block) <= set(KINDS), 'POSTCAL_TYPES')
    for kind, entry in block.items():
        need(isinstance(entry, dict) and set(entry) == {'config', 'params'}, 'POSTCAL_ENTRY')
        cfg, params = entry['config'], entry['params']
        need(isinstance(cfg, dict) and set(cfg) <= set(OPTIONS[kind]), 'POSTCAL_OPTIONS')
        need(isinstance(params, dict) and set(params) <= {'T', 'prior', 'route_T', 'route_counts', 'top'}, 'POSTCAL_PARAMS')
        need('debias' not in cfg or type(cfg['debias']) is bool, 'POSTCAL_SWITCH')
        need('fixed_top' not in cfg or type(cfg['fixed_top']) is bool, 'POSTCAL_SWITCH')
        mode = cfg.get('temperature', False)
        need(any(mode is m or (isinstance(m, str) and mode == m) for m in TEMPERATURE_MODES), 'POSTCAL_TEMPERATURE_MODE')
        if mode:
            need(_temperature_value(params.get('T')), 'POSTCAL_TEMPERATURE')
        if mode == 'route':
            cells = params.get('route_T')
            need(isinstance(cells, dict) and set(cells) <= set(ROUTE_BUCKETS)
                 and all(_temperature_value(v) for v in cells.values()), 'POSTCAL_ROUTE_TEMPERATURE')
        else:
            need('route_T' not in params, 'POSTCAL_ROUTE_TEMPERATURE_UNUSED')
        counts = params.get('route_counts')
        need(counts is None or (isinstance(counts, dict) and set(counts) <= set(ROUTE_BUCKETS)
                                and all(type(v) is int and v >= 0 for v in counts.values())), 'POSTCAL_ROUTE_COUNTS')
        if cfg.get('debias'):
            prior = params.get('prior')
            need(isinstance(prior, dict) and all(
                k.isdigit() and int(k) >= 2 and isinstance(v, list) and len(v) == int(k)
                and all(finite(w) and w > 0 for w in v) for k, v in prior.items()), 'POSTCAL_PRIOR')
        if cfg.get('floor') is not None:
            need(finite(cfg['floor']) and .8 < cfg['floor'] < 1, 'POSTCAL_FLOOR')
        if 'mode_weight' in cfg:
            need(finite(cfg['mode_weight']) and 0 <= cfg['mode_weight'] <= 1, 'POSTCAL_MODE_WEIGHT')
        residual = cfg.get('symmetric_residual', False)
        need(residual is False or residual in SYMMETRIC_RESIDUAL_MODES, 'POSTCAL_SCORE_RESIDUAL')
        if residual:
            need(not cfg.get('mode_weight') and not cfg.get('temperature'), 'POSTCAL_SCORE_RESIDUAL_EXCLUSIVE')
            need(finite(params.get('top')) and .5 < params['top'] < 1, 'POSTCAL_SCORE_RESIDUAL_TOP')
        elif not cfg.get('fixed_top'):
            need('top' not in params, 'POSTCAL_SCORE_RESIDUAL_TOP_UNUSED')
        if cfg.get('fixed_top'):
            need(kind == 'choice', 'POSTCAL_CHOICE_TOP_TYPE')
            need(finite(params.get('top')) and .5 < params['top'] < 1, 'POSTCAL_CHOICE_TOP')
    return block


def active(block):
    """{type: [active option names]} for the types this block changes."""
    return {kind: sorted(k for k, v in entry['config'].items() if v)
            for kind, entry in sorted(block.items()) if n_active(entry['config'])}


# ---------------------------------------------------------------- service hook
def wrap_service(service, block, core):
    """Calibrate each answer after the service computed it (same order as the measured setup:
    after the deadline wrapper and its item rules, before the answer is serialised)."""
    policy = validate(block)
    need(not getattr(service, '_post_calibration_wrapped', False), 'POSTCAL_ALREADY_WRAPPED')
    original = service._one_impl

    def one(task):
        answer, detail = original(task)
        kind = task['question']['type']
        entry = policy.get(kind)
        if entry is None or not n_active(entry['config']):
            return answer, detail
        labels = core.render(task)[0]
        if len(labels) < 2:
            return answer, detail                        # one option or level: nothing to calibrate
        probs = answer['probabilities']
        if set(probs) == set(labels):
            keys = list(labels)
        elif kind == 'noul' and set(probs) == {'false', 'true'}:
            keys = ['false', 'true']                     # the service's no/yes order
        else:
            raise core.ItemFailure('post-calibration label mismatch')
        bucket = route_bucket(detail)
        try:
            q = apply_one(kind, entry['config'], entry['params'], [probs[k] for k in keys], bucket)
        except (RuntimeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise core.ItemFailure('post-calibration failed') from exc
        calibrated = dict(zip(keys, q))
        answer['probabilities'] = {k: calibrated[k] for k in probs}
        by_label = dict(zip(labels, q))
        detail['prediction'] = max(labels, key=by_label.get)
        if kind == 'noul':
            answer['noul'] = float(calibrated.get('yes', calibrated.get('true')))
        elif kind == 'choice':
            answer['choice'] = detail['prediction']
        detail['post_calibration'] = dict(options=sorted(k for k, v in entry['config'].items() if v), route=bucket)
        return answer, detail

    service._one_impl = one
    service._post_calibration_wrapped = True
    return service
