"""Reproducible three-condition execution, with no access to gold in model input."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import platform
import re
import time
import importlib.metadata
from hw3_utils import (Runtime, Limits, initial_image_observations, run_baseline,
                       public_case, save_json, save_jsonl, make_annotation_rows, source_recall_at_k)
from llm_backends import ModelError


def evaluate_suite(cases, *, backend, vision_backend, retriever, chunks, sources,
                   system_prompt, agent_class, parser, output_root, k=3, limits=None,
                   configuration=None):
    """Freeze settings before this call. It can be used for development or final test.

    Every case produces a row in all conditions. Failed episodes are not dropped.
    Initial image observations are reused across conditions and their cost is saved
    separately. No automatic hidden retry and no live judge model is used.
    """
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case['id']):
            raise ValueError("Case IDs must be plain alphanumeric identifiers, not paths")
        if not isinstance(case.get('question'), str) or not case['question'].strip() or len(case['question']) > 300:
            raise ValueError("Use a nonempty question of at most 300 characters; put source content in the corpus")
        if case.get('allow_write', False) is not False:
            raise ValueError("Benchmark cases cannot grant writes. Use the separate real-user approval demonstration.")
    from setup_helpers import validate_case
    for case in cases:
        validate_case(case, sources)
    limits = limits or Limits()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)  # Do not overwrite a previous experiment.
    settings = {"configuration": configuration or {}, "backend": backend.describe(),
                "vision_backend": vision_backend.describe(), "limits": asdict(limits), "k": k,
                "system_prompt": system_prompt, "python": platform.python_version(),
                "source_hashes": {sid: s['sha256'] for sid, s in sources.items()},
                "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                "index_sha256": hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()}
    settings['packages'] = {}
    for package in ('numpy', 'scikit-learn', 'requests', 'jsonschema', 'sentence-transformers'):
        try: settings['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    save_json(root/'settings.json', settings)
    save_json(root/'cases.json', cases)
    save_json(root/'source_snapshot.json', {sid:{k:v for k,v in s.items() if k not in {'text','resolved_path'}} for sid,s in sources.items()})
    all_runs = {name: {} for name in ('direct', 'rag', 'agent')}
    preprocessing = []
    for case in cases:
        q = public_case(case)  # These are the only evaluation fields the model sees.
        shared = Runtime(backend, retriever, chunks, sources, root/'preprocess'/case['id'],
                         vision_backend=vision_backend, limits=limits)
        observations, preprocessing_failed = [], False
        try:
            observations = initial_image_observations(shared, q)
        except (ModelError, RuntimeError, ValueError):
            preprocessing_failed = True
        prep = {'case_id': q['id'], 'observations': observations, 'trace': shared.trace,
                'model_calls': shared.model_calls, 'elapsed_seconds': time.perf_counter()-shared.started,
                'failed': preprocessing_failed}
        preprocessing.append(prep)
        save_json(root/'preprocess'/f"{case['id']}.json", prep)
        for condition in all_runs:
            rt = Runtime(backend, retriever, chunks, sources, root/condition/case['id']/'artifacts',
                         vision_backend=vision_backend, allow_write=case.get('allow_write', False),
                         unavailable_tools=case.get('unavailable_tools', ()), limits=limits)
            if preprocessing_failed:
                result = rt.fail('Shared image preprocessing failed')
            elif condition == 'agent':
                result = agent_class(rt, system_prompt).run(q, initial_observations=observations)
            else:
                result = run_baseline(rt, q, system_prompt, parser, rag=condition=='rag', k=k,
                                      initial_observations=observations)
            # A shared scripted image observation also makes an episode non-empirical.
            result['mock_used'] |= any(t.get('is_mock', False) for t in shared.trace)
            result['case_id'], result['condition'] = case['id'], condition
            required = [sid for sid in case['relevant_source_ids'] if sources[sid]['kind'] == 'text']
            searches = [e['observation']['results'] for e in rt.trace
                        if e.get('event') == 'tool' and e.get('action', {}).get('tool') == 'search'
                        and e.get('observation', {}).get('ok')]
            # Agent queries are adaptive: report first-search Recall@k and also
            # cumulative source coverage separately; do not call the union top-k.
            result['retrieval'] = {
                'k': k, 'required_text_sources': required,
                'first_search_recall_at_k': (source_recall_at_k(searches[0] if searches else [], required, k)
                                             if condition != 'direct' else None),
                'cumulative_source_coverage': (len(set(required) & {h['source_id'] for hits in searches for h in hits}) / len(required)
                                               if required and condition != 'direct' else None),
                'search_calls': len(searches)}
            all_runs[condition][case['id']] = result
            save_json(root/condition/f"{case['id']}.json", result)
        print(f"Saved all conditions for {case['id']}")
    save_json(root/'preprocessing_summary.json', preprocessing)
    for name, runs in all_runs.items():
        save_jsonl(root/f'{name}_human_annotations.jsonl', make_annotation_rows(cases, runs, name))
    save_json(root/'completion.json', {'case_count':len(cases), 'episode_count':sum(len(v) for v in all_runs.values()),
                                    'conditions': list(all_runs), 'human_scoring_complete':False})
    return all_runs
