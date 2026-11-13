"""Offline package checks: integrity, model identity, configuration, adapter contract, serving
policy, calibration and disclosure items. No model is executed and nothing is downloaded.

A preparation package (config package.finalized false) and a frozen package
(finalized true plus FREEZE.json) are both checked. The freeze may change only
package.finalized in config.json and replace README.md, DISCLOSURE.md and
MODEL_CARD.md; every other file keeps the bytes of the prepared package. A Git
working copy of the package (a `.git` directory next to the files) is accepted.

SERVICE_POLICY (written by the builder from the selected serving parameters) states what this
package serves: the routing mode, threshold and quota, the drafting budget and its prompt limit, the
per-decision deadline and the item rules (types that may draft, low-margin choice re-read), the
post-calibration, the unparsed policy, the cost-fuse lines and prices and the yes/no confidence
constant of the base calibration. Every check below compares the configuration, calibration and fit
records with it, and the guards hold whatever it says.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
MODEL_KEY='metask-gemma4-31b-lora-w4a16'
MODEL_NAME='MetaDecision-31B-Rain'
ADAPTER_REF='model-subdir:adapter'
ADAPTER_SUBDIR='adapter'
ADAPTER_FILES=('adapter_config.json','adapter_model.safetensors')
TARGETS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
LORA=dict(rank=16,alpha=32,target_modules=sorted(TARGETS))
PRICING_BASE='google/gemma-4-31B-it'
TRAINING_BASE='google/gemma-4-31B-it-qat-q4_0-unquantized'
TRAINING_REVISION='1e4d8beecacb8b7590c1d8bedd7335f687bf311f'
SERVED_BASE='google/gemma-4-31B-it-qat-w4a16-ct'
SERVED_REVISION='52f3f65bc7a02d555763bc923bd1d9094898219d'
SERVICE_POLICY={'protocol': 'adapter_arm_31b_v3', 'mode': 'routed', 'quota': 0.49, 'fuse': 0.49, 'margin_threshold': 6.625023671062799, 'fit_protocol': '31b-adapter-arm-v3-routed', 'fit_n': 231, 'base_fit': {'q': 0.056, 'draft': 256}, 'noul_p_cal': 0.898, 'draft': {'max_new_tokens': 192, 'reasoning_words': 50}, 'slow_max_prompt_tokens': 840, 'unparsed_policy': 'force_answer', 'slow_excluded_types': [], 'cost_fuse': {'arm': 0.9, 'enabled': True, 'release': 0.85, 'warmup': 100}, 'cost_fuse_prices': {'input_per_million': 0.059819178473256074, 'output_per_million': 0.059819178473256074}, 'item_deadline': {'item_seconds': 9.0, 'response_reserve_s': 0.35, 'readout_reserve_init_s': 1.0, 'readout_reserve_min_s': 0.3, 'readout_safety': 1.5, 'chunk_tokens': 32, 'min_chunk_tokens': 8, 'token_safety': 1.25, 'ewma_alpha': 0.3, 'warmup_tokens': 33, 'warmup_repeats': 3}, 'item_rules': {'slow_types': ['choice', 'noul', 'score'], 'reread_tau': 4.0}, 'post_calibration': {'choice': {'config': {'debias': False, 'fixed_top': True, 'temperature': 'route'}, 'params': {'top': 0.9152542372881356, 'T': 1.0, 'route_T': {'slow': 1.5}}}, 'noul': {'config': {'floor': None, 'temperature': 'type'}, 'params': {'T': 0.6538613699913967}}, 'score': {'config': {'symmetric_residual': 'uniform', 'temperature': False}, 'params': {'top': 0.9386666666666666}}}, 'post_calibration_active': True, 'core_patch': None, 'v3_params_sha256': 'b28ed73bb5a36e99a280a47a6b846e0c5b2de54f34a96efa4a10754b71ea7573', 'threshold_rule': 'eligible_items'}
INPUT_BUDGET=6653
MAX_MODEL_LEN=6656
GPU_MEMORY_UTILIZATION=0.94
MAX_LORA_RANK=16
PROTOCOL='adapter_arm_31b_v3'
FIT_PROTOCOLS={'fast_only':'31b-adapter-arm-v3-q0','routed':'31b-adapter-arm-v3-routed'}
# The reference cost fuse of the fast-only configuration; a routed package states its own lines in SERVICE_POLICY.
COST_FUSE={'enabled':True,'warmup':100,'arm':0.85,'release':0.8}
NOUL_P_CAL_RANGE=(0.81,0.95)
SLOW_READOUT_MARGIN=32
DRAFTING_WALL_TIMEOUT_SECONDS=60
SLOW_QUOTA_MAX=0.49
BASE_FUSE=0.2
ITEM_SECONDS_MAX=9.5
KINDS=('choice','noul','score')
SERVED_MODE=(SERVICE_POLICY or {}).get('mode')
REQUIRED=('core.py backends.py quota.py attention.py model_setup.py server.py bench.py item_deadline.py post_calibration.py '
          'reconcile_usage.py launch.py serve.sh selftest.py selftest.sh pkg_check.py '
          'prepare_model.sh config.json calibration.json MODEL_MANIFEST.json ADAPTER_MANIFEST.json '
          'PIPELINE_PROVENANCE.json DISCLOSURE.md README.md MODEL_CARD.md '
          'requirements.lock requirements.measured.lock LICENSE MANIFEST.sha256').split()
RUNTIME=('core.py','backends.py','quota.py','attention.py','model_setup.py','server.py','launch.py','item_deadline.py','post_calibration.py')
REFERENCE_CODE=('core.py','backends.py','quota.py','attention.py','model_setup.py','server.py','bench.py','reconcile_usage.py')
FREEZE_SCHEMA='metadecision_31b_freeze_v1'
FREEZE_DOCUMENTS=('README.md','DISCLOSURE.md','MODEL_CARD.md')
CONFIG_CHANGE='package.finalized: false -> true (single textual replacement)'
FROZEN_FLAG,PREPARED_FLAG=b'"finalized": true',b'"finalized": false'
TEXT_SUFFIXES=('.py','.md','.json','.sh','.lock','.txt')
def percent(value):
    """'6%' for 0.06, '2.5%' for 0.025, '6.25%' for 0.0625: the service's quota and fuse lines as written in the
    documents, exact (never rounded)."""
    from decimal import Decimal
    text=format((Decimal(repr(float(value)))*100).normalize(),'f')
    return (text.rstrip('0').rstrip('.') if '.' in text else text)+'%'

def policy_disclosure(policy):
    """Policy facts DISCLOSURE.md must state: the yes/no constant; routed: the drafting pass, its quota, budget,
    prompt limit, 60 s limit and the cost-fuse lines."""
    if not policy:return ()
    items=('%.3f'%policy['noul_p_cal'],)
    if policy['mode']!='routed':return items
    pct=percent(policy['quota'])
    items+=('drafting pass','slow-path quota '+pct,'at most '+pct+' of all decisions',
            'at most %d tokens'%policy['draft']['max_new_tokens'],
            'longer than {:,} tokens'.format(policy['slow_max_prompt_tokens']),
            'deadline of %s s'%format(policy['item_deadline']['item_seconds'],'g'),'Cost fuse',
            percent(policy['cost_fuse']['arm']),percent(policy['cost_fuse']['release']))
    rules=policy.get('item_rules') or {}
    if rules.get('reread_tau'):items+=('reverse order','below %s'%format(rules['reread_tau'],'g'))
    if set(rules.get('slow_types') or KINDS)!=set(KINDS):items+=('only these question types',)
    if policy.get('post_calibration_active'):items+=('Post-calibration',)
    return items

# Necessary disclosure: base repo@revision, own weights, quantization, unmerged adapter,
# input budget and the cost/latency measurement conditions, plus the policy facts.
DISCLOSURE_REQUIRED=(PRICING_BASE,TRAINING_BASE,TRAINING_REVISION,SERVED_BASE,SERVED_REVISION,
                     'rank 16','alpha 32','W4A16','not merged','max_lora_rank=16','6,653',
                     'VLLM_BATCH_INVARIANT=1','engine token')+policy_disclosure(SERVICE_POLICY)
# README requirements: tested GPU and memory, the adapter route's vLLM options, the download tool, the mode.
README_REQUIRED=('NVIDIA A100 80GB','gpu_memory_utilization=0.94','max_model_len=6656','enable_lora',
                 'max_lora_rank=16','The model files take','`hf`','hf download')+(
                 ('`"mode": "%s"`'%SERVED_MODE,) if SERVED_MODE else ())

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def identity_sha256(m):
    """Identity of the model folder: official root files + the unmerged adapter/ files."""
    identity={'format':m['format'],'files':m['files']}
    if m.get('adapter') is not None:
        a=m['adapter']
        identity['adapter']={'format':a['format'],'subdir':a['subdir'],'files':a['files']}
    return canonical(identity)

def check_manifest(root=ROOT):
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if not re.fullmatch('[0-9a-f]{64}',digest) or Path(name).name!=name or name in expected:
            raise ValueError('Unsafe or duplicate manifest entry')
        p=root/name
        if p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Missing, linked or changed package member: '+name)
        expected[name]=digest
    actual={p.name for p in root.iterdir() if p.is_file() and p.name!='MANIFEST.sha256'}
    if actual!=set(expected):raise ValueError('Manifest membership differs')
    for p in root.iterdir():
        if p.is_symlink() or (p.is_dir() and p.name not in ('__pycache__','.git')):
            raise ValueError('Unexpected package directory or symlink')
    return expected

def front_matter(text):
    """Minimal YAML front-matter reader for top-level 'key: value' lines."""
    lines=text.splitlines()
    if not lines or lines[0].strip()!='---':raise ValueError('Model card lacks YAML front matter')
    values={}
    for line in lines[1:]:
        if line.strip()=='---':return values
        match=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*):\s*(\S.*)?',line)
        if match:values[match[1]]=(match[2] or '').strip()
    raise ValueError('Unterminated model card front matter')

def check_adapter_contract(c,m):
    adapter=m.get('adapter')
    profiles=[(p.get('model') or {}) for p in c.get('profiles',{}).values()]
    if (not isinstance(adapter,dict) or adapter.get('format')!='peft-lora' or adapter.get('subdir')!=ADAPTER_SUBDIR
            or not isinstance(adapter.get('files'),dict)):
        raise ValueError('The manifest must pin the unmerged adapter/ folder')
    if adapter['files'] and (set(adapter['files'])!=set(ADAPTER_FILES) or adapter.get('merged') is not False
            or adapter.get('rank')!=LORA['rank'] or adapter.get('alpha')!=LORA['alpha']
            or adapter.get('target_modules')!=LORA['target_modules'] or adapter.get('variant') not in ('trained_original','key_value_aligned')
            or adapter.get('training_base')!=TRAINING_BASE or adapter.get('training_revision')!=TRAINING_REVISION):
        raise ValueError('Adapter manifest differs from the trained r16/alpha32 unmerged LoRA contract')
    if c['model'].get('adapter')!=ADAPTER_REF or any('adapter' in p and p['adapter']!=ADAPTER_REF for p in profiles):
        raise ValueError('Configuration must reference the model-relative adapter')
    options=c['model'].get('vllm_options',{})
    if options.get('max_lora_rank')!=MAX_LORA_RANK or 'enable_lora' in options:
        raise ValueError('vLLM max_lora_rank=16 required; enable_lora is backend-owned')
    record=read('ADAPTER_MANIFEST.json')
    if (sha(ROOT/'ADAPTER_MANIFEST.json')!=adapter.get('adapter_manifest_sha256') or record.get('files')!=adapter['files']
            or record.get('variant')!=adapter.get('variant') or (record.get('lora') or {}).get('merged') is not False
            or (record.get('served_base') or {}).get('repo_id')!=SERVED_BASE
            or (record.get('served_base') or {}).get('revision')!=SERVED_REVISION):
        raise ValueError('ADAPTER_MANIFEST.json does not bind the pinned adapter files')

def policy_problems(policy):
    """Guards that hold whatever the builder wrote: mode, quota, yes/no constant, drafting budget and context."""
    if not isinstance(policy,dict):return ['service policy pending']
    problems=[]
    p=policy.get('noul_p_cal')
    if isinstance(p,bool) or not isinstance(p,float) or not NOUL_P_CAL_RANGE[0]<=p<=NOUL_P_CAL_RANGE[1]:
        problems.append('yes/no constant outside [0.81, 0.95]')
    if policy.get('protocol')!=PROTOCOL or policy.get('fit_protocol')!=FIT_PROTOCOLS.get(policy.get('mode')):
        problems.append('protocol or fit protocol')
    q=policy.get('quota')
    if policy.get('mode')=='routed':
        problems+=v4_policy_problems(policy)
    if policy.get('mode')=='fast_only':
        if q!=0 or policy.get('cost_fuse')!=COST_FUSE or policy.get('slow_excluded_types'):
            problems.append('fast-only policy with a quota, other cost fuse or excluded types')
    elif policy.get('mode')=='routed':
        fuse,draft=policy.get('cost_fuse') or {},policy.get('draft') or {}
        if isinstance(q,bool) or not isinstance(q,float) or not 0<q<=SLOW_QUOTA_MAX:problems.append('quota')
        if (set(fuse)!={'enabled','warmup','arm','release'} or fuse['enabled'] is not True or fuse['warmup']!=100
                or not 0.5<=fuse['release']<fuse['arm']<=0.9):problems.append('cost fuse lines')
        length,prompt=draft.get('max_new_tokens'),policy.get('slow_max_prompt_tokens')
        if (type(length) is not int or type(prompt) is not int or type(draft.get('reasoning_words')) is not int
                or prompt+length+SLOW_READOUT_MARGIN>MAX_MODEL_LEN or prompt>INPUT_BUDGET):
            problems.append('drafting budget outside the vLLM context')
        if policy.get('unparsed_policy') not in ('fast','force_answer'):problems.append('unparsed policy')
        if policy.get('slow_excluded_types') not in ([],['noul']):problems.append('excluded types')
        if bool(policy.get('slow_excluded_types'))!=(policy.get('core_patch') is not None):
            problems.append('excluded slow types need exactly the v3 core patch')
    else:
        problems.append('mode')
    return problems

def v4_policy_problems(policy):
    """Fuse, cost-fuse prices, per-decision deadline, item rules and post-calibration of a routed package."""
    problems=[]
    q=policy.get('quota')
    if not isinstance(q,float) or policy.get('fuse')!=max(BASE_FUSE,q):problems.append('fuse must be max(0.2, quota)')
    t=policy.get('margin_threshold')
    if isinstance(t,bool) or not isinstance(t,float) or not 0<=t<float('inf'):problems.append('margin threshold')
    prices=policy.get('cost_fuse_prices') or {}
    if (set(prices)!={'input_per_million','output_per_million'}
            or any(isinstance(v,bool) or not isinstance(v,float) or not 0<v<float('inf') for v in prices.values())):
        problems.append('cost-fuse prices')
    try:
        import item_deadline
        s=item_deadline.settings_from(policy.get('item_deadline'))
        if s!=policy.get('item_deadline') or not 0<s['item_seconds']<=ITEM_SECONDS_MAX:problems.append('item deadline')
    except Exception:
        problems.append('item deadline')
    rules=policy.get('item_rules')
    if (not isinstance(rules,dict) or set(rules)!={'slow_types','reread_tau'} or not isinstance(rules['slow_types'],list)
            or not rules['slow_types'] or not set(rules['slow_types'])<=set(KINDS)
            or len(set(rules['slow_types']))!=len(rules['slow_types'])
            or isinstance(rules['reread_tau'],bool) or not isinstance(rules['reread_tau'],float)
            or not 0<=rules['reread_tau']<float('inf')):
        problems.append('item rules')
    try:
        import post_calibration
        block=post_calibration.validate(policy.get('post_calibration'))
        if bool(post_calibration.active(block))!=bool(policy.get('post_calibration_active')):problems.append('post-calibration flag')
    except Exception:
        problems.append('post-calibration')
    return problems

def check_serving_policy(c):
    """SERVICE_POLICY's mode, quota, draft, prompt limit and cost fuse; 6653 input budget everywhere server.py can
    apply it; 31B context/memory; the served configuration is server.py's fixed point."""
    from core import apply_profile, apply_budget_tier
    problems=policy_problems(SERVICE_POLICY)
    if problems:raise ValueError('Service policy invalid: '+', '.join(problems))
    policy=SERVICE_POLICY
    r,limits,options,g=c['routing'],c['limits'],c['model'].get('vllm_options',{}),c['generation']
    q,draft=policy['quota'],policy['draft']
    if (r['mode']!=policy['mode'] or r['quota']!=q or r['target_slow_share']!=q or c['bench'].get('target_slow_share')!=q
            or r['fuse']!=policy['fuse'] or c['usage']!={'basis':'engine'} or r.get('cost_fuse')!=policy['cost_fuse']):
        raise ValueError('Changed serving mode, quota, usage or cost-fuse convention')
    if (r['margin_threshold']!=policy['margin_threshold']
            or c['pricing']['models'][c['pricing']['model_key']]!=policy['cost_fuse_prices']
            or c.get('item_deadline')!=policy['item_deadline'] or c.get('item_rules')!=policy['item_rules']
            or c.get('post_calibration')!=policy['post_calibration']):
        raise ValueError('Changed threshold, cost-fuse prices, deadline, item rules or post-calibration')
    if (r['margin_threshold']!=0 if policy['mode']=='fast_only' else not 0<r['margin_threshold']<float('inf')):
        raise ValueError('Changed serving threshold')
    if ((g['max_new_tokens'],g['reasoning_words'])!=(draft['max_new_tokens'],draft['reasoning_words'])
            or (r['tiers']['hard'].get('max_new_tokens'),r['tiers']['hard'].get('reasoning_words'))!=(draft['max_new_tokens'],draft['reasoning_words'])
            or r['slow_max_prompt_tokens']!=policy['slow_max_prompt_tokens'] or r['unparsed_policy']!=policy['unparsed_policy']
            or r.get('slow_excluded_types',[])!=policy['slow_excluded_types']
            or ('slow_excluded_types' in r and not r['slow_excluded_types'])):
        raise ValueError('Changed drafting budget, prompt limit, unparsed policy or excluded types')
    if policy['mode']=='routed' and (r.get('long_item_slow_policy')!='fast_only'
                                     or g.get('wall_timeout_seconds')!=DRAFTING_WALL_TIMEOUT_SECONDS):
        raise ValueError('Changed routed serving, drafting timeout or long-item policy')
    if (limits['max_prompt_tokens'],limits['physical_max_prompt_tokens'])!=(INPUT_BUDGET,INPUT_BUDGET) \
            or any(r['tiers'][t]['max_prompt_tokens']!=INPUT_BUDGET for t in ('normal','long')) \
            or limits['oom_retry_prompt_tokens']>INPUT_BUDGET:
        raise ValueError('Input budget must be 6653 for every prompt limit')
    if options.get('max_model_len')!=MAX_MODEL_LEN or options.get('gpu_memory_utilization')!=GPU_MEMORY_UTILIZATION:
        raise ValueError('vLLM context 6656 and GPU memory 0.94 required')
    served=copy.deepcopy(c)
    apply_profile(served)
    apply_budget_tier(served,None)
    served['routing']['mode']=SERVED_MODE
    if served!=c:
        raise ValueError('server.py would apply a different configuration than the measured one')
    import core
    if policy['slow_excluded_types'] and getattr(core,'SLOW_TYPE_EXCLUSION',None)!='routing.slow_excluded_types':
        raise ValueError('Excluded slow types need the patched core.py')

def check_calibration_policy(cal):
    """The yes/no constant of SERVICE_POLICY in the fast, slow and fallback blocks; drafting-pass and fallback
    answers use the first-pass temperatures (stated in DISCLOSURE.md)."""
    blocks=cal['models'][MODEL_KEY]
    p=SERVICE_POLICY['noul_p_cal']
    for side in ('fast','slow','fallback'):
        if blocks[side]['noul_calibration']['p_cal']!=p or blocks[side]['by_type']['noul'].get('p_cal')!=p:
            raise ValueError('Calibration block does not carry the stated yes/no constant: '+side)
        if blocks[side]['temperature_by_kind']!=blocks['fast']['temperature_by_kind']:
            raise ValueError('DISCLOSURE.md states that drafting-pass answers use the first-pass calibration')

def prepared_config_bytes(data):
    """Undo the freeze's single textual change of config.json."""
    if data.count(FROZEN_FLAG)!=1 or PREPARED_FLAG in data:
        raise ValueError('Frozen config.json must differ from the prepared one only in package.finalized')
    return data.replace(FROZEN_FLAG,PREPARED_FLAG)

def check_freeze(c,m,members):
    """None for a preparation package; the verified FREEZE.json record for a frozen one."""
    path=ROOT/'FREEZE.json'
    if not c['package'].get('finalized'):
        if path.exists():raise ValueError('FREEZE.json in a package that is not finalized')
        return None
    if not path.is_file():raise ValueError('Finalized package without FREEZE.json')
    f=read('FREEZE.json')
    if f.get('schema')!=FREEZE_SCHEMA or f.get('status')!='FROZEN':
        raise ValueError('Unknown freeze record')
    if not c['package'].get('nonformal_fitted') or c['package'].get('sample_only'):
        raise ValueError('Only the fitted package can be frozen')
    if not re.fullmatch('[0-9a-f]{40}',str(m.get('revision') or '')) or m['repo_id'].startswith('OWNER/'):
        raise ValueError('Frozen package lacks the own repository and full revision')
    if (f.get('repo_id'),f.get('revision'),f.get('model_identity_sha256'))!=(m['repo_id'],m['revision'],m['identity_sha256']):
        raise ValueError('Freeze record names another repository, revision or model')
    prepared=f.get('prepared_files')
    replaced=f.get('replaced_documents') or {}
    if not isinstance(prepared,dict) or set(members)-{'FREEZE.json'}!=set(prepared) or set(replaced)-set(FREEZE_DOCUMENTS):
        raise ValueError('Frozen package membership differs from the prepared package')
    for name,digest in members.items():
        if name=='FREEZE.json':continue
        if name=='config.json':
            before=hashlib.sha256(prepared_config_bytes((ROOT/name).read_bytes())).hexdigest()
            if before!=prepared[name] or f.get('config')!={'prepared_sha256':before,'frozen_sha256':digest,'change':CONFIG_CHANGE}:
                raise ValueError('Frozen config.json is not the recorded freeze of the prepared config')
        elif name in replaced:
            if replaced[name]!={'prepared_sha256':prepared[name],'frozen_sha256':digest}:
                raise ValueError('Replaced document differs from the freeze record: '+name)
        elif digest!=prepared[name]:
            raise ValueError('Freeze changed a file other than config.json and the documents: '+name)
    return f

def disclosure_problems(root=ROOT):
    """Necessary disclosure items that DISCLOSURE.md (and the model card base_model) must state."""
    text=(root/'DISCLOSURE.md').read_text(encoding='utf-8',errors='replace')
    problems=['DISCLOSURE.md lacks: '+item for item in DISCLOSURE_REQUIRED if item not in text]
    readme=(root/'README.md').read_text(encoding='utf-8',errors='replace')
    problems+=['README.md lacks: '+item for item in README_REQUIRED if item not in readme]
    card=(root/'MODEL_CARD.md').read_text(encoding='utf-8',errors='replace')
    try:
        values=front_matter(card)
    except ValueError as exc:
        return problems+[str(exc)]
    if values.get('base_model')!=PRICING_BASE:
        problems.append('MODEL_CARD.md front matter base_model must be '+PRICING_BASE)
    for item in (TRAINING_BASE,TRAINING_REVISION,SERVED_BASE,SERVED_REVISION):
        if item not in card:problems.append('MODEL_CARD.md lacks: '+item)
    return problems

def checks():
    for name in REQUIRED:
        if not (ROOT/name).is_file():raise ValueError('Missing '+name)
    members=check_manifest()
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    c,m,prov=read('config.json'),read('MODEL_MANIFEST.json'),read('PIPELINE_PROVENANCE.json')
    from core import validate_config, validate_threshold, validate_calibration
    import launch
    validate_config(c)
    if c['model']['model_key']!=MODEL_KEY or c['model']['name']!=MODEL_NAME or set(c['profiles'])!={MODEL_KEY}:
        raise ValueError('Wrong trained model key')
    if c['pipeline']['arm']!='trained' or prov.get('arm')!='trained' or m.get('arm')!='trained':
        raise ValueError('Arm differs between config, manifest and provenance')
    if c['pipeline']['model_identity_sha256']!=m['identity_sha256']:
        raise ValueError('Model identity mismatch')
    if m.get('revision') is not None and not re.fullmatch('[0-9a-f]{40}',m['revision']):
        raise ValueError('Abbreviated/mutable revision rejected')
    if m['repo_id'].startswith('google/'):
        raise ValueError('Own HF repository required')
    if (m.get('served_base')or{}).get('repo_id')!=SERVED_BASE or (m.get('served_base')or{}).get('revision')!=SERVED_REVISION:
        raise ValueError('Served base must be the official W4A16 CT revision')
    check_adapter_contract(c,m)
    check_serving_policy(c)
    frozen=check_freeze(c,m,members)
    complete=bool(m['files']) and bool(m['adapter']['files'])
    if complete and identity_sha256(m)!=m['identity_sha256']:raise ValueError('Manifest identity hash differs')
    if not complete and m['identity_sha256'] is not None:raise ValueError('Incomplete model has an identity')
    cal=read('calibration.json')
    if set(cal['models'])!={MODEL_KEY}:raise ValueError('Foreign calibration model')
    for side in ('fast','slow','fallback'):validate_calibration(cal,MODEL_KEY,side)
    check_calibration_policy(cal)
    fitted=c['package'].get('nonformal_fitted',False)
    if fitted:
        if validate_threshold(c):raise ValueError('Threshold identity differs from the configuration')
        # The served configuration is the materialized one; its threshold identity must still hold.
        if validate_threshold(launch.materialize_adapter(c,Path('/materialized/adapter'))):
            raise ValueError('Materialized threshold identity differs')
        lock=read('SERVICE_LOCK.json')
        report=read('fit_report.json')
        if (lock['protocol']!=prov.get('protocol') or lock['protocol']!=PROTOCOL
                or lock['serving_identity_sha256']!=m['identity_sha256'] or lock['arm']!='trained'
                or lock.get('adapter_files')!=m['adapter']['files'] or lock.get('q')!=SERVICE_POLICY['quota']):
            raise ValueError('Fit lock identity mismatch')
        actual={'config_file':frozen['config']['prepared_sha256'] if frozen else sha(ROOT/'config.json'),
                'calibration':sha(ROOT/'calibration.json'),'fit_report':sha(ROOT/'fit_report.json')}
        for stem,digest in actual.items():
            if lock[stem+'_sha256']!=digest:raise ValueError('Fit hash mismatch')
        base=SERVICE_POLICY['base_fit']
        if (report.get('fit_protocol')!=SERVICE_POLICY['fit_protocol'] or report.get('n')!=SERVICE_POLICY['fit_n']
                or report.get('q')!=base['q']
                or (report.get('noul_constant') or {}).get('value')!=SERVICE_POLICY['noul_p_cal']
                or report.get('draft')!=base['draft']):
            raise ValueError('Wrong fitting method')
    else:
        if not c['package']['sample_only'] or not cal['models'][MODEL_KEY].get('placeholder'):
            raise ValueError('Unfitted package must clearly carry placeholders')
    if (c['package']['finalized'] and frozen is None) or prov.get('release_ready') or prov.get('official_submission_authorized'):
        raise ValueError('Only the recorded freeze may finalize; the builder never authorizes a release')
    for name in REFERENCE_CODE+('item_deadline.py','post_calibration.py'):
        if sha(ROOT/name)!=prov['source_files'][name]:raise ValueError('Request code changed: '+name)
    patch_record=prov.get('core_patch')
    if patch_record!=SERVICE_POLICY.get('core_patch'):raise ValueError('Core patch record differs from the policy')
    if patch_record is None:
        if prov['source_files']['core.py']!=prov['reference_source_files']['core.py']:raise ValueError('Reference core.py changed')
    else:
        raise ValueError('This package serves the unchanged core.py')
    server_patch=prov.get('server_patch') or {}
    if (sha(ROOT/'server.py')!=server_patch.get('patched_server_sha256')
            or prov['reference_source_files']['server.py']!=server_patch.get('reference_server_sha256')):
        raise ValueError('server.py is not the declared extension patch of the reference server.py')
    if any(prov['source_files'][n]!=prov['reference_source_files'][n] for n in REFERENCE_CODE if n not in ('core.py','server.py')):
        raise ValueError('Reference request code changed')
    if 'install_extensions(service, c)' not in (ROOT/'server.py').read_text(encoding='utf-8'):
        raise ValueError('server.py does not install the serving extensions')
    for name in RUNTIME:
        text=(ROOT/name).read_text(encoding='utf-8')
        if re.search(r'\brmtree\s*\(|\b(?:snapshot_download|hf_hub_download)\s*\(',text):
            raise ValueError('Runtime downloads or recursive deletion found')
    launch_text=(ROOT/'launch.py').read_text(encoding='utf-8')
    for token in ("HF_HUB_OFFLINE='1'","TRANSFORMERS_OFFLINE='1'","env['VLLM_BATCH_INVARIANT']='1'","env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'",
                  "VLLM_USE_FLASHINFER_SAMPLER='0'","def check_adapter(","def materialize_adapter(",
                  "    check_adapter(model)\n","cfg=write_runtime_config(startup,runtime)","config.runtime.json",
                  "SERVED_MODE='%s'"%SERVED_MODE,"mode=SERVED_MODE","(ROOT/'FREEZE.json').is_file()"):
        if token not in launch_text:raise ValueError('Offline/invariant/adapter/mode/freeze startup setting missing: '+token.strip())
    if launch.ADAPTER_REF!=ADAPTER_REF or launch.MODEL_KEY!=MODEL_KEY or launch.SERVED_MODE!=c['routing']['mode']:
        raise ValueError('Launcher adapter reference, model key or mode differs')
    for p in ROOT.glob('*.sh'):
        if re.search(r'\brm\s+-[^\n]*r|\brmtree\b',p.read_text(encoding='utf-8')):
            raise ValueError('Recursive deletion in shell')
    serve=(ROOT/'serve.sh').read_text(encoding='utf-8')
    if 'VLLM_USE_FLASHINFER_SAMPLER=0' not in serve or 'VLLM_BATCH_INVARIANT=1' not in serve or 'VLLM_WORKER_MULTIPROC_METHOD=spawn' not in serve:
        raise ValueError('serve.sh must set VLLM_USE_FLASHINFER_SAMPLER=0, VLLM_BATCH_INVARIANT=1 and VLLM_WORKER_MULTIPROC_METHOD=spawn')
    problems=disclosure_problems()
    if problems:raise ValueError('Necessary disclosure incomplete: '+'; '.join(problems))
    pins=dict(line.split('==',1) for line in (ROOT/'requirements.lock').read_text().splitlines() if '==' in line and not line.startswith('#'))
    for name,version in launch.VERSIONS.items():
        if pins.get(name,'').split('+')[0]!=version:raise ValueError('Runtime version pin mismatch')
    return {'status':'FROZEN_PACKAGE_CHECK_PASS' if frozen else 'PREPARATION_CHECK_PASS','model_execution':False,'network':False,
            'adapter':'unmerged adapter/ ('+m['adapter'].get('variant','pending')+')','served_mode':SERVED_MODE,
            'input_budget':INPUT_BUDGET,'service_fit':'frozen' if frozen else 'fitted_not_frozen' if fitted else 'pending',
            'serving_extensions':['item_deadline','item_rules','post_calibration']}

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(checks(),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
