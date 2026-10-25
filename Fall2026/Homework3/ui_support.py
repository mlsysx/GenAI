"""Provided safety/serialization utilities for the OPTIONAL Q6 UI.

These functions make no model calls. They do not implement the student's UI
integration, progress summary, or session snapshot/restore functions. Local JSON
is trusted only as a learner's own saved record, never as write authorization.
This teaching scaffold is not a multi-user security service.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_ROOT = PROJECT_ROOT / "outputs"
MAX_STATE_BYTES = 16_000_000
SCHEMA_VERSION = 1


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def new_ui_state(goal_texts):
    if not isinstance(goal_texts, list) or len(goal_texts) < 2:
        raise ValueError("Record at least two actual learning goals")
    if any(not isinstance(g, str) or not g.strip() or len(g) > 800 for g in goal_texts):
        raise ValueError("Each goal must be a nonempty string of at most 800 characters")
    return {"goals": [{"id": f"G{i+1}", "text": g.strip()} for i, g in enumerate(goal_texts)],
            "completed_activities": [], "confidence_ratings": []}


def validate_ui_state(value, *, turn_count=None):
    if not isinstance(value, dict) or set(value) != {"goals", "completed_activities", "confidence_ratings"}:
        raise ValueError("Unexpected UI progress fields")
    if not isinstance(value['goals'], list) or len(value['goals']) < 2:
        raise ValueError("At least two goals are required")
    ids=[]
    for goal in value['goals']:
        if not isinstance(goal,dict) or set(goal)!={'id','text'}:
            raise ValueError("A goal has id and text, not a model-assigned mastery score")
        if not isinstance(goal['id'],str) or not isinstance(goal['text'],str) or not goal['text'].strip():
            raise ValueError("Invalid goal")
        ids.append(goal['id'])
    if len(set(ids))!=len(ids):raise ValueError("Goal IDs must be unique")
    if not isinstance(value['completed_activities'],list) or not isinstance(value['confidence_ratings'],list):
        raise ValueError("Progress entries must be lists")
    for a in value['completed_activities']:
        if not isinstance(a,dict) or set(a)!={'label','turn','origin','recorded_at'}:
            raise ValueError("Invalid completed-activity record")
        if a['origin']!='learner_confirmed' or not isinstance(a['label'],str) or not a['label'].strip():
            raise ValueError("Completion is learner-confirmed activity, not verified mastery")
        if type(a['turn']) is not int or a['turn']<1 or (turn_count is not None and a['turn']>turn_count):
            raise ValueError("An activity must refer to an actual interaction turn")
    for c in value['confidence_ratings']:
        if not isinstance(c,dict) or set(c)!={'goal_id','rating','origin','recorded_at'}:
            raise ValueError("Invalid confidence record")
        if c['goal_id'] not in ids or type(c['rating']) is not int or not 1<=c['rating']<=5 or c['origin']!='learner_reported':
            raise ValueError("Confidence must be learner-reported on the stated 1–5 scale")
    return copy.deepcopy(value)


def record_completed_activity(ui_state, label, turn, *, turn_count):
    if not isinstance(label,str) or not label.strip() or type(turn) is not int or not 1<=turn<=turn_count:
        raise ValueError("Select a real interaction and a nonempty activity label")
    ui_state['completed_activities'].append({'label':label.strip(),'turn':turn,
        'origin':'learner_confirmed','recorded_at':now_utc()})


def record_confidence(ui_state, goal_id, rating):
    if goal_id not in {g['id'] for g in ui_state['goals']} or type(rating) is not int or not 1<=rating<=5:
        raise ValueError("Use an existing goal and integer rating 1–5")
    ui_state['confidence_ratings'].append({'goal_id':goal_id,'rating':rating,
        'origin':'learner_reported','recorded_at':now_utc()})


def active_seconds(session):
    """Exclude human waiting after a model response; preserve accumulated work."""
    end = session._returned_at if session._returned_at is not None else time.perf_counter()
    return max(0.0, end - session.runtime.started)


def session_binding(session, config):
    """Bind a local snapshot to the same corpus, chunk inventory and model settings."""
    rt=session.runtime
    cfg_keys=('domain','chunk_max_words','chunk_overlap','retriever','top_k','embedding_model','embedding_revision')
    chunk_records=[{k:c[k] for k in ('id','source_id','text','start_char','end_char') if k in c}
                   for _,c in sorted(rt.chunks.items())]
    material={
        'source_hashes':{sid:s['sha256'] for sid,s in sorted(rt.sources.items())},
        'source_metadata':{sid:{k:v for k,v in s.items() if k not in {'text','resolved_path'}} for sid,s in sorted(rt.sources.items())},
        'chunk_hash':hashlib.sha256(json.dumps(chunk_records,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'answer_model':rt.backend.describe(),'vision_model':rt.vision_backend.describe(),
        'limits':asdict(rt.limits),'max_turns':session.max_turns,
        'configuration':{k:config.get(k) for k in cfg_keys},
    }
    return {'sha256':hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
            'source_hashes':material['source_hashes']}


def progress_path(path, *, root=STATE_ROOT):
    root=Path(root).resolve();p=Path(path)
    if not p.is_absolute():p=PROJECT_ROOT/p
    p=p.resolve()
    if not p.is_relative_to(root) or p.suffix.lower()!='.json':
        raise ValueError("Store progress as a .json file inside the approved outputs directory")
    return p


def write_progress_json(path, value, *, root=STATE_ROOT):
    """Atomic host-initiated checkpoint. Not a model-callable report-writing tool."""
    p=progress_path(path,root=root)
    payload=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(payload)>MAX_STATE_BYTES:raise ValueError("Progress record is too large; explicitly archive, do not truncate silently")
    p.parent.mkdir(parents=True,exist_ok=True)
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=p.parent,delete=False) as f:
            name=f.name;f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(name,p)
    finally:
        if name and Path(name).exists():Path(name).unlink()
    return p


def read_progress_json(path, *, root=STATE_ROOT):
    p=progress_path(path,root=root)
    if p.stat().st_size>MAX_STATE_BYTES:raise ValueError("Progress record exceeds the maximum size")
    try:
        return json.loads(p.read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
    except (UnicodeDecodeError,json.JSONDecodeError) as e:
        raise ValueError('Invalid UTF-8 JSON progress record') from e


def validate_snapshot(snapshot, session, config):
    """Validate completely BEFORE mutating a freshly created LearningSession.

    This does not authenticate editable local files. Load only your own records;
    stored approval is never imported as an active permission.
    """
    if not isinstance(snapshot,dict):raise ValueError('Expected a progress object')
    required={'schema_version','saved_at','binding','learner_state','history','turns','ui_state',
              'pending_artifact','closed','runtime','prior_wall_seconds'}
    if set(snapshot)!=required or snapshot['schema_version']!=SCHEMA_VERSION:
        raise ValueError('Unsupported or unexpected progress schema')
    if snapshot['binding']!=session_binding(session,config):
        raise ValueError('Model, corpus, chunking, limits or retrieval configuration changed; start an explicitly separate session')
    state=snapshot['learner_state']
    state_fields={'topic','goal','prior_knowledge','pending_question','attempts','unresolved_questions','evidence_ids'}
    if not isinstance(state,dict) or set(state)!=state_fields:raise ValueError('Invalid learner state')
    if any(not isinstance(state[k],str) for k in ('topic','goal','prior_knowledge')):raise ValueError('Invalid learner profile')
    if state['pending_question'] is not None and not isinstance(state['pending_question'],str):raise ValueError('Invalid pending question')
    for k in ('attempts','unresolved_questions','evidence_ids'):
        if not isinstance(state[k],list):raise ValueError('Invalid learner-state list')
    for attempt in state['attempts']:
        if not isinstance(attempt,dict) or set(attempt)!={'question','learner_reply','feedback','feedback_verified'}:
            raise ValueError('Invalid attempt record')
        if not isinstance(attempt['question'],str) or not isinstance(attempt['learner_reply'],str):raise ValueError('Invalid recorded attempt')
        if attempt['feedback'] is not None and not isinstance(attempt['feedback'],str):raise ValueError('Invalid feedback')
        if attempt['feedback_verified'] is not False:raise ValueError('The starter does not assign verified mastery or verified feedback')
    if not all(isinstance(q,str) for q in state['unresolved_questions']):raise ValueError('Invalid question list')
    if not isinstance(snapshot['history'],list) or not isinstance(snapshot['turns'],list):raise ValueError('Invalid history/turns')
    if len(snapshot['turns'])>session.max_turns:raise ValueError('Recorded turn budget exceeded')
    if any(not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in {'user','assistant'} or not isinstance(m['content'],str) for m in snapshot['history']):
        raise ValueError('Saved history may contain only actual user/assistant messages, not system-role overrides')
    if type(snapshot['closed']) is not bool:raise ValueError('Invalid session-closed flag')
    validate_ui_state(snapshot['ui_state'],turn_count=len(snapshot['turns']))
    rt=snapshot['runtime']
    fields={'steps','model_calls','tool_calls','active_seconds','trace','observed_ids','signatures','artifacts'}
    if not isinstance(rt,dict) or set(rt)!=fields:raise ValueError('Runtime permissions must not be serialized as restorable grants')
    for k,limit in [('steps',session.runtime.limits.max_steps),('model_calls',session.runtime.limits.max_model_calls),('tool_calls',session.runtime.limits.max_tool_calls)]:
        if type(rt[k]) is not int or not 0<=rt[k]<=limit:raise ValueError('Invalid resource counters')
    for value in [rt['active_seconds'],snapshot['prior_wall_seconds']]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid recorded duration')
    for key in ('trace','observed_ids','signatures','artifacts'):
        if not isinstance(rt[key],list):raise ValueError('Invalid runtime history')
    if any(not isinstance(e,dict) for e in rt['trace']):raise ValueError('Invalid trace event')
    if not all(isinstance(i,str) for i in rt['observed_ids']+rt['signatures']+state['evidence_ids']):raise ValueError('Invalid evidence/signature IDs')
    known = session.runtime.known_ids | calculation_evidence_ids(rt['trace'])
    if not set(rt['observed_ids']).issubset(known) or not set(state['evidence_ids']).issubset(known):raise ValueError('Unknown saved evidence ID')
    artifact=snapshot['pending_artifact']
    if artifact is not None:
        from hw3_utils import TOOL_SPECS, check_schema
        check_schema(artifact,TOOL_SPECS['save_report']['parameters'])
    return copy.deepcopy(snapshot)


def calculation_evidence_ids(trace):
    """Recover derived evidence IDs from successful arithmetic receipts only."""
    import re
    from hw3_utils import safe_arithmetic
    result = set()
    for event in trace:
        if event.get('event') != 'tool' or event.get('action', {}).get('tool') != 'calculate':
            continue
        obs = event.get('observation', {})
        if obs.get('ok') is not True:
            continue
        eid = obs.get('id', '')
        expression = event.get('action', {}).get('arguments', {}).get('expression')
        if (not isinstance(eid, str) or not re.fullmatch(r'CALC:[0-9]{3,}', eid)
                or obs.get('expression') != expression
                or type(obs.get('value')) not in (int, float)
                or obs['value'] != safe_arithmetic(expression)):
            raise ValueError('Invalid saved arithmetic receipt')
        result.add(eid)
    return result


def archive_approval_events(trace):
    """Keep audit history, but do not expose old approvals as current receipts."""
    result=copy.deepcopy(trace)
    for e in result:
        if e.get('event')=='host_approval':
            e['event']='archived_host_approval';e['active_authorization']=False
    return result


def artifact_digest(artifact):
    return hashlib.sha256(artifact['content'].encode('utf-8')).hexdigest()


def record_artifact_review(session, *, expected_filename, expected_digest, approved):
    """HOST UI button callback only; verify the user reviewed the current proposal."""
    if type(approved) is not bool:raise ValueError('Use the actual UI decision')
    a=session.pending_artifact
    if a is None or a['filename']!=expected_filename or artifact_digest(a)!=expected_digest:
        raise ValueError('The proposed report changed; display it again before approval')
    digest=session.runtime.approve_report(a['filename'],a['content'],confirmed=approved)
    session.history.append({'role':'user','content':f'Actual host review: {"approved" if approved else "denied"} {a["filename"]}; content SHA256 {digest}.'})
    return digest
