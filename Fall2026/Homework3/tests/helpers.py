"""Explicit OFFLINE fixtures, never final demonstrations or benchmark results."""
import copy
import json
from pathlib import Path
from hw3_utils import Runtime, Limits, TfidfRetriever, load_corpus, final_action, validate_action_object
from llm_backends import ScriptedBackend
from learning_session import LearningSession
from ui_support import SCHEMA_VERSION, now_utc, session_binding, new_ui_state
ROOT = Path(__file__).resolve().parents[1]

# OFFLINE software cases only; students create their own evaluation cases.
CONCEPT_CASE = {'id': 'demo_01', 'category': 'concept', 'question': 'What is source-level Recall@k, and how are duplicate chunks from one source counted?', 'image_ids': [], 'expected_status': 'answered', 'relevant_source_ids': ['DOC01'], 'answer_criteria': ['Defines distinct required sources retrieved divided by all required sources', 'Counts a repeated source once']}
VISUAL_CASE = {'id': 'demo_02', 'category': 'visual', 'question': 'In this toy chart, what is dense Recall@2, and how much higher is it than lexical Recall@2?', 'image_ids': ['IMG01'], 'expected_status': 'answered', 'relevant_source_ids': ['IMG01'], 'answer_criteria': ['Dense Recall@2 is 0.75', 'Lexical Recall@2 is 0.50; difference 0.25 or 25 percentage points', 'Identifies the numbers as a toy example, not published performance']}



def fixture_index():
    sources = load_corpus(ROOT/'data/demo/manifest.json')
    # Whole-document fixture chunks only; not a solution to student chunking.
    chunks = [dict(id=s['id']+':fixture', source_id=s['id'], text=s['text'],
                   start_char=0, end_char=len(s['text']), title=s['title'], date=s['date'], status=s['status'])
              for s in sources.values() if s['kind']=='text']
    return sources, chunks, TfidfRetriever(chunks)


def runtime(out, replies=(), **kwargs):
    sources, chunks, retriever = fixture_index()
    return Runtime(ScriptedBackend(replies), retriever, chunks, sources, out, **kwargs)


def tool(name, **arguments):
    return {'action':'tool', 'tool':name, 'arguments':arguments}


def parser(raw, specs=None):
    return validate_action_object(json.loads(raw))


class FinalOnlyFixture:
    """Single scripted final per turn, used to test the HOST, never an agent implementation."""
    def __init__(self, rt): self.runtime=rt
    def run(self, case, initial_observations=(), user_history=()):
        return self.runtime.finish(parser(self.runtime.next_decision([{'role':'user','content':case['question']}])) )


def session(out, finals=()):
    return LearningSession(FinalOnlyFixture(runtime(out,[json.dumps(f) for f in finals])),
                           topic='Test topic',goal='First goal; second goal',prior_knowledge='Beginner')


def snapshot(s, config=None):
    # Empty-session fixture; student persistence is tested separately.
    return {'schema_version':SCHEMA_VERSION,'saved_at':now_utc(),'binding':session_binding(s,config or {}),
            'learner_state':copy.deepcopy(s.state),'history':copy.deepcopy(s.history),'turns':copy.deepcopy(s.turns),
            'ui_state':new_ui_state(['First goal','Second goal']), 'pending_artifact':s.pending_artifact,
            'closed':s.closed,'prior_wall_seconds':0.0,
            'runtime':{'steps':s.runtime.steps,'model_calls':s.runtime.model_calls,'tool_calls':s.runtime.tool_calls,
                       'active_seconds':0.0,'trace':copy.deepcopy(s.runtime.trace),
                       'observed_ids':sorted(s.runtime.observed_ids),'signatures':sorted(s.runtime.signatures),
                       'artifacts':copy.deepcopy(s.runtime.artifacts)}}
