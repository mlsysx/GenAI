"""Eight Q5 contract tests. Expected to ERROR on the uncompleted starter."""
import json
import tempfile
import unittest
import numpy as np
from hw3_utils import final_action, Limits
from student_agent import chunk_document, DenseRetriever, parse_action, build_system_prompt, EvidenceAgent
from tests.helpers import fixture_index, runtime, tool

class StudentTests(unittest.TestCase):
    def test_chunk_exact_spans_overlap_and_end(self):
        source={'id':'S','title':'S','date':'2026','status':'current','text':'alpha  beta\ngamma delta epsilon'}
        chunks=chunk_document(source,max_words=3,overlap=1)
        self.assertEqual([c['text'] for c in chunks],['alpha  beta\ngamma','gamma delta epsilon'])
        self.assertEqual(len({c['id'] for c in chunks}),2)
        self.assertEqual(chunks,chunk_document(source,max_words=3,overlap=1))
        for c in chunks:self.assertEqual(c['text'],source['text'][c['start_char']:c['end_char']])
        self.assertEqual(chunk_document(dict(source,text='   ')),[])
        with self.assertRaises(ValueError):chunk_document(source,max_words=3,overlap=3)

    def test_dense_cosine_not_dot_and_current_filter(self):
        chunks=[{'id':'a','text':'A','status':'current'}, {'id':'b','text':'B','status':'current'},
                {'id':'old','text':'OLD','status':'superseded'},{'id':'zero','text':'ZERO','status':'current'}]
        vectors={'A':[100,100],'B':[1,0],'OLD':[1,0],'ZERO':[0,0],'query':[1,0]}
        r=DenseRetriever(chunks,lambda texts:[vectors[t] for t in texts])
        hits=r.search('query',k=2)
        self.assertEqual([h['id'] for h in hits],['b','a'])
        self.assertAlmostEqual(hits[0]['score'],1)
        self.assertTrue(all(np.isfinite(h['score']) for h in r.search('query',k=10)))
        self.assertEqual(r.search(''),[])

    def test_dense_ties_are_stable(self):
        r=DenseRetriever([{'id':'z','text':'Z','status':'current'},{'id':'a','text':'A','status':'current'}],lambda texts:[[1,0] for _ in texts])
        self.assertEqual([h['id'] for h in r.search('q',2)],['a','z'])

    def test_parser_schema_and_no_prose(self):
        f=final_action('answered','Example')
        self.assertEqual(parse_action(json.dumps(f)),f)
        for raw in ['prefix '+json.dumps(f),json.dumps(f)+json.dumps(f),'{"action":"shell"}',json.dumps(tool('search',query='q',k=99))]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse_action(raw)

    def test_system_prompt_contract(self):
        p=build_system_prompt('Test domain')
        self.assertIsInstance(p,str);self.assertIn('Test domain',p)
        for name in ['search','read_chunk','inspect_image','calculate','save_report','awaiting_answer','needs_approval']:
            self.assertIn(name,p)
        self.assertGreater(len(p),400)

    def test_agent_uses_actual_tool_observation_without_gold(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,[json.dumps(tool('calculate',expression='2+3')),json.dumps(final_action('answered','Five'))])
            result=EvidenceAgent(rt,'OFFLINE fixture prompt').run({'id':'x','question':'Calculate','answer_criteria':['GOLD_SECRET']})
            self.assertEqual(result['final']['status'],'answered');self.assertEqual(rt.tool_calls,1)
            self.assertTrue(any(e.get('observation',{}).get('value')==5 for e in rt.trace))
            second=json.dumps(rt.backend.calls[1]['messages'])
            self.assertIn('CALC:001',second);self.assertNotIn('GOLD_SECRET',second)

    def test_agent_repairs_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,['malformed']*5,limits=Limits(max_repairs=1))
            result=EvidenceAgent(rt,'OFFLINE').run({'id':'x','question':'Hi'})
            self.assertEqual(result['final']['status'],'failed');self.assertLessEqual(rt.model_calls,2)

    def test_agent_continuation_keeps_runtime_budget(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,[json.dumps(final_action('answered','One'))],limits=Limits(max_model_calls=1))
            agent=EvidenceAgent(rt,'OFFLINE')
            self.assertEqual(agent.run({'id':'x','question':'First'})['final']['status'],'answered')
            self.assertEqual(agent.run({'id':'y','question':'Second'})['final']['status'],'failed')
            self.assertEqual(rt.model_calls,1)
