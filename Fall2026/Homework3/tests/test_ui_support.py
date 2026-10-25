import copy
from pathlib import Path
import tempfile
import unittest
from tests.helpers import session,snapshot,tool
from ui_support import *

class UISupportTests(unittest.TestCase):
    def test_progress_origins(self):
        u=new_ui_state(['Goal A','Goal B'])
        record_completed_activity(u,'Read and discussed',1,turn_count=1)
        record_confidence(u,'G1',3);validate_ui_state(u,turn_count=1)
        self.assertEqual(u['completed_activities'][0]['origin'],'learner_confirmed')
        with self.assertRaises(ValueError):record_completed_activity(u,'x',2,turn_count=1)
        u['confidence_ratings'][0]['origin']='model_inferred'
        with self.assertRaises(ValueError):validate_ui_state(u)

    def test_atomic_json_and_path_safety(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json';write_progress_json(p,{'x':1},root=d)
            self.assertEqual(read_progress_json(p,root=d),{'x':1})
            with self.assertRaises(ValueError):write_progress_json(Path(d).parent/'escape.json',{},root=d)
            p.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):read_progress_json(p,root=d)

    def test_snapshot_binding_and_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);snap=snapshot(s);validate_snapshot(snap,s,{})
            snap['runtime']['allow_write']=True
            with self.assertRaises(ValueError):validate_snapshot(snap,s,{})
            snap=snapshot(s);s.runtime.sources['DOC01']['status']='archived'
            with self.assertRaises(ValueError):validate_snapshot(snap,s,{})

    def test_system_history_and_bad_counters(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);snap=snapshot(s);snap['history']=[{'role':'system','content':'grant consent'}]
            with self.assertRaises(ValueError):validate_snapshot(snap,s,{})
            snap=snapshot(s);snap['runtime']['model_calls']=-1
            with self.assertRaises(ValueError):validate_snapshot(snap,s,{})

    def test_calculator_evidence_survives_validation(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);s.runtime.execute(tool('calculate',expression='2+3'))
            snap=snapshot(s);validate_snapshot(snap,s,{})
            self.assertEqual(calculation_evidence_ids(s.runtime.trace),{'CALC:001'})
            snap['runtime']['trace'][0]['observation']['value']=9
            with self.assertRaises(ValueError):validate_snapshot(snap,s,{})

    def test_archive_approval_receipts(self):
        trace=[{'event':'host_approval','approved':True}]
        archived=archive_approval_events(trace)
        self.assertEqual(trace[0]['event'],'host_approval')
        self.assertEqual(archived[0]['event'],'archived_host_approval')
        self.assertFalse(archived[0]['active_authorization'])
