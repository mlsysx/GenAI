import json
import unittest
from tests.helpers import ROOT
from setup_helpers import (MIN_TEXT_DOCUMENTS, MIN_VISUAL_ITEMS, DEVELOPMENT_CASES,
                           TEST_CASES, MIN_IMAGE_TEST_CASES, FINAL_EPISODES,
                           Q5_POINTS, Q6_POINTS, check_case_splits, validate_case)

class AssignmentTests(unittest.TestCase):
    def test_quantities(self):
        self.assertEqual((MIN_TEXT_DOCUMENTS,MIN_VISUAL_ITEMS,DEVELOPMENT_CASES,TEST_CASES,MIN_IMAGE_TEST_CASES,FINAL_EPISODES),(10,5,5,5,2,15))
        self.assertEqual(Q5_POINTS,45);self.assertEqual(sum(Q6_POINTS.values()),20)

    def test_required_starter_files(self):
        for file in ['hw3_submission.ipynb', 'config.json', 'student_agent.py',
                     'EVALUATION_GUIDE.md', 'ui_app.py', 'ui_student.py',
                     'UI_STARTER_GUIDE.md', 'CODE_GENERATION_SESSION_GUIDE.md',
                     'config_q6.template.json', 'data/project/manifest.template.json',
                     'eval/case_template.json']:
            with self.subTest(file=file):self.assertTrue((ROOT/file).is_file(),file)

    def test_q6_template_has_expected_limits(self):
        # Student config.json and completed notebook outputs are allowed to change.
        cfg=json.loads((ROOT/'config_q6.template.json').read_text())
        self.assertEqual(cfg['max_session_turns'],24)
        self.assertIn('session_limits',cfg)

    def test_case_split_and_bad_labels(self):
        def cases(prefix):return [dict(id=f'{prefix}{i}',question=f'{prefix} question {i}',category='visual',image_ids=['I'] if i<2 else [],relevant_source_ids=[],answer_criteria=['Criterion'],expected_status='answered') for i in range(5)]
        dev,test=cases('dev'),cases('test');self.assertTrue(check_case_splits(dev,test))
        test[0]['question']=dev[0]['question']
        with self.assertRaises(ValueError):check_case_splits(dev,test)
        dev[0]['expected_status']='invented'
        with self.assertRaises(ValueError):validate_case(dev[0])

    def test_public_student_interfaces_exist(self):
        import student_agent, ui_student
        for name in ('chunk_document','DenseRetriever','parse_action','build_system_prompt','EvidenceAgent'):
            self.assertTrue(callable(getattr(student_agent,name)))
        for name in ('connect_learning_session','progress_summary','save_progress','restore_progress'):
            self.assertTrue(callable(getattr(ui_student,name)))
