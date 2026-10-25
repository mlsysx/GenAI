"""Aggregate manually scored real experiment logs; NEVER calls a model or fills judgments."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import statistics
from hw3_utils import load_jsonl, save_json, summarize_annotations

CONDITIONS = ('direct', 'rag', 'agent')


def token_totals(events):
    model_events = [e for e in events if e.get('event') == 'model']
    def count(field):
        values = [(e.get('usage') or {}).get(field) for e in model_events]
        # No calls means zero tokens; a call with missing usage means unknown.
        return sum(values) if all(type(v) is int and v >= 0 for v in values) else None
    return {f: count(f) for f in ('input_tokens', 'output_tokens')}


def summarize(root):
    root = Path(root)
    completion = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    expected_count = completion['case_count']
    if completion['episode_count'] != expected_count * 3:
        raise ValueError('Incomplete three-condition experiment')
    prep = {x['case_id']: x for x in json.loads((root/'preprocessing_summary.json').read_text(encoding='utf-8'))}
    report = {'note': 'Human scoring from saved real runs only. Shared visual preprocessing is added to each stand-alone condition; do not sum those figures as the experiment invoice.',
              'shared_preprocessing_actual_seconds': sum(p['elapsed_seconds'] for p in prep.values()),
              'conditions': {}}
    for condition in CONDITIONS:
        rows = load_jsonl(root/f'{condition}_human_annotations.jsonl')
        if len(rows) != expected_count or len({x['case_id'] for x in rows}) != expected_count:
            raise ValueError('Annotation rows must cover all cases exactly once')
        metrics = summarize_annotations(rows)  # Rejects mocks, blanks, and credited failed runs.
        resources = []
        for row in rows:
            cid = row['case_id']
            run = json.loads((root/condition/f'{cid}.json').read_text(encoding='utf-8'))
            if run.get('mock_used'):
                raise ValueError('A scripted episode cannot be summarized as an empirical run')
            pre = prep[cid]
            events = pre['trace'] + run['trace']
            resources.append({'case_id':cid,
                'model_calls_including_initial_vision':run['metrics']['model_calls']+pre['model_calls'],
                'tool_calls_including_initial_vision':run['metrics']['tool_calls']+sum(t.get('event')=='tool' for t in pre['trace']),
                'seconds_including_initial_vision':run['metrics']['elapsed_seconds']+pre['elapsed_seconds'],
                'tool_errors_including_initial_vision':sum(t.get('event')=='tool' and not t.get('observation',{}).get('ok',False) for t in events),
                'token_usage':token_totals(events)})
        retrieval = [json.loads((root/condition/f"{row['case_id']}.json").read_text())['retrieval'] for row in rows]
        recalls = [r['first_search_recall_at_k'] for r in retrieval if r['first_search_recall_at_k'] is not None]
        metrics['first_search_recall_at_k'] = {'n': len(recalls), 'mean': statistics.mean(recalls) if recalls else None}
        audits = [json.loads((root/condition/f"{row['case_id']}.json").read_text())['citation_audit'] for row in rows]
        cited = sum(len(a['cited_ids']) for a in audits)
        invalid = sum(sum(i in set(a['unknown_ids'] + a['unobserved_ids']) for i in a['cited_ids']) for a in audits)
        metrics['citation_id_validity'] = {'valid': cited-invalid, 'total': cited, 'rate': (cited-invalid)/cited if cited else None}
        metrics['resources'] = resources
        metrics['total_model_calls'] = sum(r['model_calls_including_initial_vision'] for r in resources)
        metrics['total_tool_calls'] = sum(r['tool_calls_including_initial_vision'] for r in resources)
        metrics['total_seconds_including_initial_vision'] = sum(r['seconds_including_initial_vision'] for r in resources)
        metrics['median_seconds_including_initial_vision'] = statistics.median(r['seconds_including_initial_vision'] for r in resources)
        metrics['monetary_cost'] = None  # Provider-specific pricing must be documented by the student.
        report['conditions'][condition] = metrics
    save_json(root/'human_scored_summary.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    result=summarize(args.run_directory)
    for name,m in result['conditions'].items():
        print(f"{name}: success {m['task_success']:.1%} ({m['n']} cases), "
              f"model calls {m['total_model_calls']}, "
              f"median seconds {m['median_seconds_including_initial_vision']:.2f}")
    print('Saved human_scored_summary.json; citations still depend on your source checks.')

if __name__=='__main__': main()
