"""Nine optional Q6 contracts. Expected to ERROR until ui_student.py is completed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hw3_utils import final_action
from tests.helpers import session, tool
from ui_support import STATE_ROOT,new_ui_state,read_progress_json,write_progress_json
import ui_student as student

class UIStudentTests(unittest.TestCase):
    def setUp(self):
        STATE_ROOT.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=STATE_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'progress.json'
        self.s=session(self.temp.name,[final_action('awaiting_answer','Try',question='Q?'),final_action('answered','Feedback')])
        self.u=new_ui_state(['First goal','Second goal']);self.cfg={}

    def test_connect_reuses_q5(self):
        cfg={'enable_live':True,'retriever':'lexical','answer_backend':{'provider':'ollama','model':'fixture'}}
        sentinel=object()
        with patch.object(student,'read_config',return_value=cfg),patch.object(student,'build_index',return_value=({},[],sentinel,None)),patch.object(student,'make_learning_session',return_value=self.s) as make:
            s,c=student.connect_learning_session('x.json',topic='Code Generation',goals=['First goal','Second goal'],output_dir=str(Path(self.temp.name)/'artifacts'))
            self.assertIs(s,self.s);self.assertEqual(c,cfg)
            self.assertIs(make.call_args.args[3],sentinel)
            self.assertEqual(s.runtime.model_calls,0)

    def test_connect_disabled_config_rejected(self):
        with patch.object(student,'read_config',return_value={'enable_live':False}),self.assertRaises(ValueError):
            student.connect_learning_session('x.json',topic='Code Generation',goals=['A','B'])

    def test_summary_actual_attempts_only(self):
        self.s.respond('Practice');self.s.respond('My answer',kind='answer')
        p=student.progress_summary(self.s,self.u)
        self.assertEqual(p['independent_attempts'],1);self.assertEqual(p['ai_feedback_entries'],1)
        self.assertEqual(p['turns'],2);self.assertEqual(p['resources']['model_calls'],2)
        self.assertEqual(p['confidence_ratings'],[]);self.assertNotIn('mastery',p)

    def test_save_json_bound_without_permissions(self):
        self.s.respond('Practice');student.save_progress(self.s,self.u,self.cfg,self.path)
        snap=read_progress_json(self.path)
        self.assertEqual(snap['learner_state']['pending_question'],'Q?')
        self.assertNotIn('allow_write',snap['runtime']);self.assertNotIn('_approved_reports',snap['runtime'])
        self.assertIn('binding',snap)

    def test_restore_pending_and_budget_no_replay(self):
        self.s.respond('Practice');student.save_progress(self.s,self.u,self.cfg,self.path)
        fresh=session(self.temp.name);u=student.restore_progress(fresh,self.cfg,self.path)
        self.assertEqual(u,self.u);self.assertEqual(fresh.state,self.s.state)
        self.assertEqual(fresh.runtime.model_calls,1);self.assertEqual(fresh.runtime.backend.calls,[])
        self.assertEqual(fresh.turns,self.s.turns)

    def test_restore_revokes_approvals(self):
        self.s.runtime.approve_report('x.md','Exact',confirmed=True)
        student.save_progress(self.s,self.u,self.cfg,self.path)
        fresh=session(self.temp.name);student.restore_progress(fresh,self.cfg,self.path)
        self.assertFalse(fresh.runtime.allow_write);self.assertFalse(fresh.runtime._approved_reports)
        self.assertFalse(any(e.get('event')=='host_approval' for e in fresh.runtime.trace))
        self.assertFalse(fresh.runtime.execute(tool('save_report',filename='x.md',content='Exact'))['ok'])

    def test_restore_rejects_changed_binding_before_mutation(self):
        student.save_progress(self.s,self.u,self.cfg,self.path)
        fresh=session(self.temp.name);before=copy.deepcopy(fresh.state)
        with self.assertRaises(ValueError):student.restore_progress(fresh,{'top_k':6},self.path)
        self.assertEqual(fresh.state,before);self.assertEqual(fresh.runtime.model_calls,0)

    def test_restore_preserves_calculator_evidence(self):
        self.s.runtime.execute(tool('calculate',expression='2+3'))
        student.save_progress(self.s,self.u,self.cfg,self.path)
        fresh=session(self.temp.name);student.restore_progress(fresh,self.cfg,self.path)
        self.assertIn('CALC:001',fresh.runtime.known_ids);self.assertIn('CALC:001',fresh.runtime.observed_ids)
        self.assertEqual(fresh.runtime.tool_calls,1)

    def test_restore_closed_session(self):
        self.s.closed=True;student.save_progress(self.s,self.u,self.cfg,self.path)
        fresh=session(self.temp.name);student.restore_progress(fresh,self.cfg,self.path)
        self.assertTrue(fresh.closed)
        with self.assertRaises(RuntimeError):fresh.respond('Should not run')
