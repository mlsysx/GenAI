import hashlib
from pathlib import Path
import tempfile
import unittest
from hw3_utils import final_action
from tests.helpers import runtime,tool,session
from ui_support import artifact_digest,record_artifact_review

class ApprovalTests(unittest.TestCase):
    def test_exact_bytes_single_use_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d);a=tool('save_report',filename='plan.md',content='Exact plan')
            self.assertFalse(rt.execute(a)['ok'])
            rt.approve_report('plan.md','Exact plan',confirmed=True)
            self.assertFalse(rt.execute(tool('save_report',filename='plan.md',content='Changed'))['ok'])
            self.assertFalse(rt.execute(tool('save_report',filename='other.md',content='Exact plan'))['ok'])
            result=rt.execute(a);self.assertTrue(result['ok'])
            self.assertEqual((Path(d)/'plan.md').read_text(),'Exact plan')
            self.assertEqual(result['artifact']['sha256'],hashlib.sha256(b'Exact plan').hexdigest())
            self.assertFalse(rt.execute(a)['ok'])

    def test_coarse_flag_is_not_approval(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d,allow_write=True)
            self.assertFalse(rt.execute(tool('save_report',filename='x.md',content='x'))['ok'])

    def test_denial_revokes_exact_grant(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d);rt.approve_report('x.md','x',confirmed=True)
            rt.approve_report('x.md','x',confirmed=False)
            self.assertFalse(rt.execute(tool('save_report',filename='x.md',content='x'))['ok'])
            self.assertFalse((Path(d)/'x.md').exists())

    def test_model_cannot_approve_or_escape(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(d)
            self.assertFalse(rt.execute(tool('approve_report',filename='x.md',content='x',confirmed=True))['ok'])
            self.assertFalse(rt.execute(tool('save_report',filename='../x.md',content='x'))['ok'])
            with self.assertRaises(ValueError):rt.approve_report('x.md','x',confirmed='yes')

    def test_stale_ui_preview(self):
        with tempfile.TemporaryDirectory() as d:
            s=session(d);s.pending_artifact={'filename':'x.md','content':'x'}
            digest=artifact_digest(s.pending_artifact);s.pending_artifact['content']='changed'
            with self.assertRaises(ValueError):record_artifact_review(s,expected_filename='x.md',expected_digest=digest,approved=True)
            self.assertFalse(s.runtime.allow_write)

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir();(root/'outside.md').write_text('untouched')
            (root/'reports/x.md').symlink_to(root/'outside.md')
            rt=runtime(root/'reports');rt.approve_report('x.md','changed',confirmed=True)
            self.assertFalse(rt.execute(tool('save_report',filename='x.md',content='changed'))['ok'])
            self.assertEqual((root/'outside.md').read_text(),'untouched')
