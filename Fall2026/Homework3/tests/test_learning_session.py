import json
import tempfile
import unittest
from unittest.mock import patch
from hw3_utils import final_action
from tests.helpers import session

class LearningSessionTests(unittest.TestCase):
    def test_actual_attempt_feedback_and_budget_persist(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d,[final_action('awaiting_answer','Try it',question='2+2?'),final_action('answered','Your answer 5 is off by one.')])
            s.respond('Practice please');self.assertEqual(s.state['pending_question'],'2+2?')
            s.respond('5',kind='answer')
            self.assertEqual(s.state['attempts'][0]['learner_reply'],'5')
            self.assertIn('5',s.state['attempts'][0]['feedback'])
            self.assertFalse(s.state['attempts'][0]['feedback_verified'])
            self.assertEqual(s.runtime.model_calls,2)
            self.assertEqual(s.turns[1]['state_before']['pending_question'],'2+2?')

    def test_solution_request_is_not_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d,[final_action('awaiting_answer','Try',question='Q?'),final_action('answered','Solution')])
            s.respond('Practice');s.respond('Please show solution',kind='request_solution')
            self.assertEqual(s.state['attempts'],[]);self.assertIsNone(s.state['pending_question'])

    def test_bad_attempt_does_not_mutate_time_or_state(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);started=s.runtime.started
            with self.assertRaises(ValueError):s.respond('answer',kind='answer')
            self.assertEqual(s.runtime.started,started);self.assertEqual(s.turns,[])

    def test_failure_closes_session(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d,[final_action('failed','OFFLINE failure')]);s.respond('Hello')
            self.assertTrue(s.closed)
            with self.assertRaises(RuntimeError):s.respond('Again')

    def test_profile_and_unresolved_are_host_owned(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);s.set_profile(topic='RAG',goal='Compare retrieval')
            s.note_unresolved('How do citations work?')
            self.assertEqual(s.state['topic'],'RAG');self.assertEqual(len(s.state['unresolved_questions']),1)
            with self.assertRaises(ValueError):s.set_profile(topic=42)

    def test_log_has_configuration_and_hashes(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            s=session(d,[final_action('answered','OFFLINE')]);s.respond('Hello');s.save_log(Path(d)/'log.json')
            data=json.loads((Path(d)/'log.json').read_text())
            self.assertTrue(data['mock_used']);self.assertIn('DOC01',data['source_hashes'])
