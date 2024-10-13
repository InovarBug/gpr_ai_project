
import unittest
from gpr_ai_no_gui import AdvancedGPRAI

class TestAdvancedGPRAI(unittest.TestCase):
    def setUp(self):
        self.ai = AdvancedGPRAI()
        self.ai.db_path = ':memory:'  # Use in-memory database for testing
        self.ai._create_db()

    def test_add_and_load_pattern(self):
        pattern = {'data': [1, 2, 3]}
        self.ai.add_pattern('Test Pattern', pattern, 'This is a test pattern')
        self.ai.load_patterns()
        self.assertEqual(len(self.ai.patterns_db), 1)
        self.assertEqual(self.ai.patterns_db[1]['name'], 'Test Pattern')
        self.assertEqual(self.ai.patterns_db[1]['description'], 'This is a test pattern')
        self.assertEqual(self.ai.patterns_db[1]['pattern'], pattern)

    def test_delete_pattern(self):
        pattern = {'data': [1, 2, 3]}
        self.ai.add_pattern('Test Pattern', pattern, 'This is a test pattern')
        self.ai.load_patterns()
        self.ai.delete_pattern(1)
        self.ai.load_patterns()
        self.assertEqual(len(self.ai.patterns_db), 0)

    def test_update_pattern(self):
        pattern = {'data': [1, 2, 3]}
        self.ai.add_pattern('Test Pattern', pattern, 'This is a test pattern')
        self.ai.load_patterns()
        self.ai.update_pattern(1, 'Updated Pattern', 'This is an updated test pattern')
        self.ai.load_patterns()
        self.assertEqual(self.ai.patterns_db[1]['name'], 'Updated Pattern')
        self.assertEqual(self.ai.patterns_db[1]['description'], 'This is an updated test pattern')

if __name__ == '__main__':
    unittest.main()
