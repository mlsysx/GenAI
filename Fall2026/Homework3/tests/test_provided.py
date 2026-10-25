import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from hw3_utils import *
from llm_backends import ChatBackend, ModelError, ScriptedBackend
from experiment import evaluate_suite
from tests.helpers import fixture_index, runtime, tool, parser, ROOT, CONCEPT_CASE, VISUAL_CASE


class ProvidedTests(unittest.TestCase):
    def test_corpus_hashes(self):
        sources, chunks, _ = fixture_index()
        self.assertEqual(len(sources), 7)
        validate_chunks(chunks, sources)
        chunks[0]['text'] += 'tampered'
        with self.assertRaises(ValueError): validate_chunks(chunks,sources)

    def test_manifest_tamper_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'s.md').write_text('hello')
            source=dict(id='S',title='S',kind='text',path='s.md',source='fixture',permission='fixture',date='2026',status='current',sha256=sha256(p/'s.md'))
            save_json(p/'m.json',{'sources':[source]})
            self.assertEqual(load_corpus(p/'m.json')['S']['text'],'hello')
            (p/'s.md').write_text('different')
            with self.assertRaises(ValueError):load_corpus(p/'m.json')
            source['path']='../outside.md';save_json(p/'m.json',{'sources':[source]})
            with self.assertRaises(ValueError):load_corpus(p/'m.json')

    def test_lexical_current_filter(self):
        _,_,r=fixture_index()
        self.assertTrue(r.search('source recall',3))
        self.assertNotIn('DOC05',{h['source_id'] for h in r.search('retrieval instructions',6)})
        self.assertEqual(r.search(''),[])

    def test_source_recall(self):
        hits=[{'source_id':'A'},{'source_id':'A'},{'source_id':'B'}]
        self.assertEqual(source_recall_at_k(hits,['A','B'],2),0.5)
        self.assertEqual(source_recall_at_k(hits,['A','B'],3),1)
        self.assertIsNone(source_recall_at_k(hits,[],3))

    def test_arithmetic(self):
        self.assertEqual(safe_arithmetic('(3+5)/2'),4)
        for expr in ['__import__("os")','2**100','True','[1][0]','1e100','x+1']:
            with self.subTest(expr=expr),self.assertRaises((ValueError,SyntaxError)):safe_arithmetic(expr)
        with self.assertRaises(ZeroDivisionError):safe_arithmetic('1/0')

    def test_action_schema(self):
        validate_action_object(final_action('awaiting_answer','Try this.',question='What is recall?'))
        for a in [tool('shell',cmd='echo hi'),tool('read_chunk',chunk_id='../x'),tool('calculate',expression='1',extra=True)]:
            if a.get('tool')=='read_chunk': continue  # Paths are rejected by runtime allowlist.
            with self.assertRaises(ValueError):validate_action_object(a)
        with self.assertRaises(ValueError):validate_action_object(final_action('needs_approval','Approve?',question='Yes?'))

    def test_tool_allowlist_repeats_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d)
            self.assertFalse(rt.execute(tool('read_chunk',chunk_id='../secrets'))['ok'])
            act=tool('calculate',expression='1+2')
            self.assertTrue(rt.execute(act)['ok'])
            self.assertFalse(rt.execute(act)['ok'])
            self.assertEqual(rt.tool_calls,3)

    def test_citations_must_be_observed(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d);cid=next(iter(rt.chunks))
            f=final_action('answered','claim',[{'claim':'claim','evidence_ids':[cid]}])
            with self.assertRaises(ValueError):rt.finish(f)
            rt.execute(tool('read_chunk',chunk_id=cid))
            self.assertFalse(rt.finish(f)['citation_audit']['support_verified'])

    def test_pixels_and_observation_label(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,['OFFLINE observation: text is unclear'])
            obs=rt.execute(tool('inspect_image',image_id='IMG01',question='What is visible?'))
            self.assertTrue(obs['ok']);self.assertTrue(obs['evidence']['generated'])
            self.assertEqual(Path(rt.backend.calls[0]['images'][0]).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')

    def test_context_and_call_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,['ok'],limits=Limits(max_model_calls=1))
            rt.next_decision([{'role':'user','content':'hi'}])
            with self.assertRaises(BudgetExceeded):rt.next_decision([{'role':'user','content':'again'}])
            self.assertEqual(rt.model_calls,1);self.assertEqual(rt.steps,1)
            rt=runtime(d,limits=Limits(max_context_chars=3))
            with self.assertRaises(BudgetExceeded):rt.next_decision([{'role':'user','content':'long'}])
            self.assertEqual(rt.model_calls,0)

    def test_public_case_strips_gold(self):
        case={'id':'x','question':'hello','image_ids':[],'answer_criteria':['GOLD_SECRET'],'expected_status':'answered'}
        self.assertNotIn('GOLD_SECRET',json.dumps(public_case(case)))

    def test_hosted_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            sources,chunks,r=fixture_index();sources['DOC01']['allow_external']=False
            b=ChatBackend('gemini','example',enable_live=False)
            with self.assertRaises(PermissionError):Runtime(b,r,chunks,sources,d)

    def test_disabled_backend_does_not_request(self):
        with patch('llm_backends.requests.post') as post:
            with self.assertRaises(ModelError):ChatBackend('ollama','fixture').generate([{'role':'user','content':'x'}])
            post.assert_not_called()

    def test_mocked_provider_payloads(self):
        replies={'huggingface':{'choices':[{'message':{'content':'{}'}}]},
                 'gemini':{'candidates':[{'content':{'parts':[{'text':'{}'}]}}]},
                 'ollama':{'message':{'content':'{}'}}}
        for provider,response in replies.items():
            with self.subTest(provider=provider),patch.dict('os.environ',{'HF_TOKEN':'OFFLINE_TEST_KEY','GEMINI_API_KEY':'OFFLINE_TEST_KEY'}),patch('llm_backends.requests.post') as post:
                post.return_value=Mock(status_code=200);post.return_value.json.return_value=response
                b=ChatBackend(provider,'fixture',enable_live=True)
                self.assertEqual(b.generate([{'role':'user','content':'inspect'}],images=[ROOT/'data/demo/images/IMG01.png']).text,'{}')
                body=json.dumps(post.call_args.kwargs['json'])
                self.assertIn('iVBOR',body)
                self.assertNotIn('OFFLINE_TEST_KEY',post.call_args.args[0])
                self.assertNotIn('OFFLINE_TEST_KEY',json.dumps(b.describe()))
                self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_provider_error_redacted(self):
        with patch('llm_backends.requests.post') as post:
            post.return_value=Mock(status_code=401,text='SECRET')
            with self.assertRaises(ModelError) as ctx:ChatBackend('ollama','x',enable_live=True).generate([{'role':'user','content':'x'}])
            self.assertNotIn('SECRET',str(ctx.exception))

    def test_suite_shares_one_image_and_preserves_failures(self):
        from tests.helpers import FinalOnlyFixture
        class Agent(FinalOnlyFixture):
            def __init__(self,rt,prompt):super().__init__(rt)
            def run(self,case,initial_observations=(),user_history=()):
                self.runtime.seed_observations(initial_observations)
                return super().run(case,initial_observations,user_history)
        cases=[copy.deepcopy(VISUAL_CASE)]
        with tempfile.TemporaryDirectory() as d:
            sources,chunks,r=fixture_index()
            b=ScriptedBackend([json.dumps(final_action('answered','OFFLINE test'))]*3)
            v=ScriptedBackend(['OFFLINE visual observation'])
            runs=evaluate_suite(cases,backend=b,vision_backend=v,retriever=r,chunks=chunks,sources=sources,
                system_prompt='OFFLINE',agent_class=Agent,parser=parser,output_root=Path(d)/'run')
            self.assertEqual(len(v.calls),1);self.assertEqual(len(b.calls),3)
            self.assertTrue(all(x['demo_02']['mock_used'] for x in runs.values()))
            self.assertTrue((Path(d)/'run/cases.json').is_file())
            b=ScriptedBackend([]);v=ScriptedBackend([ModelError('OFFLINE failure')])
            runs=evaluate_suite(cases,backend=b,vision_backend=v,retriever=r,chunks=chunks,sources=sources,
                system_prompt='OFFLINE',agent_class=Agent,parser=parser,output_root=Path(d)/'failed')
            self.assertTrue(all(x['demo_02']['final']['status']=='failed' for x in runs.values()))

    def test_human_annotations_cannot_be_mock_or_missing(self):
        rows=make_annotation_rows([copy.deepcopy(CONCEPT_CASE)],{},'direct')
        with self.assertRaises(ValueError):summarize_annotations(rows)
        rows[0].update(task_success=1,mock_used=True)
        with self.assertRaises(ValueError):summarize_annotations(rows)
