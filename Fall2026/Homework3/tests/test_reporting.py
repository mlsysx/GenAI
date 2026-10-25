import copy
import json
from pathlib import Path
import tempfile
import unittest
from hw3_utils import make_annotation_rows,summarize_annotations,final_action,save_json,save_jsonl
from summarize_run import summarize,token_totals

class ReportingTests(unittest.TestCase):
    def row(self):
        return dict(case_id='c',condition='direct',category='concept',task_success=1,mock_used=False,failed_or_missing=False,
                    supported_citations=0,total_citations=0,supported_factual_claims=0,total_factual_claims=0,
                    appropriate_uncertainty='N/A',visual_grounding='N/A',error_stage='none')

    def test_no_claims_is_not_perfect_precision(self):
        r=summarize_annotations([self.row()]);self.assertIsNone(r['citation_support_precision'])
        self.assertEqual(r['appropriate_uncertainty']['n'],0)

    def test_partial_annotation_rejected(self):
        row=self.row();row['supported_citations']=None
        with self.assertRaises(ValueError):summarize_annotations([row])
        row=self.row();row['appropriate_uncertainty']=None
        with self.assertRaises(ValueError):summarize_annotations([row])

    def test_unknown_usage_not_zero(self):
        self.assertEqual(token_totals([])['input_tokens'],0)
        self.assertIsNone(token_totals([{'event':'model'}])['input_tokens'])

    def test_three_condition_summary(self):
        # Fabricated measurement structure for SOFTWARE TEST ONLY; no model findings.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            save_json(root/'completion.json',{'case_count':1,'episode_count':3})
            save_json(root/'preprocessing_summary.json',[{'case_id':'c','elapsed_seconds':0,'model_calls':0,'trace':[]}])
            for name in ('direct','rag','agent'):
                save_jsonl(root/f'{name}_human_annotations.jsonl',[dict(self.row(),condition=name)])
                save_json(root/name/'c.json',{'mock_used':False,'final':final_action('answered','software fixture'),
                    'metrics':{'model_calls':1,'tool_calls':0,'elapsed_seconds':1},'trace':[{'event':'model','usage':{'input_tokens':2,'output_tokens':3}}],
                    'retrieval':{'first_search_recall_at_k':None if name=='direct' else 0.5},
                    'citation_audit':{'cited_ids':[],'unknown_ids':[],'unobserved_ids':[]}})
            r=summarize(root)
            self.assertEqual(r['conditions']['rag']['first_search_recall_at_k']['mean'],0.5)
            self.assertIsNone(r['conditions']['direct']['citation_id_validity']['rate'])
