import unittest
import re
from pathlib import Path

class EmptyStr:
    def split(self, sep=None):
        return []

def load_arxiv():
    source = Path('papersolver/commands.py').read_text()
    match = re.search(r'class Arxiv\(Command\):(.*?)(?=class PaperReplace)', source, re.S)
    code = match.group(0)
    class Command:
        pass
    def extract_prompt(text, word):
        pattern = rf"```{word}(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL)
        if not blocks:
            return EmptyStr()
        return "\n".join(blocks).strip()
    namespace = {'Command': Command, 'extract_prompt': extract_prompt, 'EmptyStr': EmptyStr}
    exec(code, namespace)
    return namespace['Arxiv']

class ParseCommandTest(unittest.TestCase):
    def setUp(self):
        Arxiv = load_arxiv()
        self.arxiv = Arxiv.__new__(Arxiv)

    def test_summary(self):
        success, data = self.arxiv.parse_command('```SUMMARY\nquery\n```')
        self.assertTrue(success)
        self.assertEqual(data, ('SUMMARY', ['query']))

    def test_full_text(self):
        success, data = self.arxiv.parse_command('```FULL_TEXT\npaper123\n```')
        self.assertTrue(success)
        self.assertEqual(data, ('FULL_TEXT', ['paper123']))

if __name__ == '__main__':
    unittest.main()
