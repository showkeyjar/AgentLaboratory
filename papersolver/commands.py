import random
import string
from utils import *
from tools.common import ArxivSearch
from copy import copy
from inference import *
from pathlib import Path
from copy import deepcopy
from common_imports import *
from agents.base import get_score
from abc import abstractmethod
from contextlib import contextmanager
import sys, os

class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        pass

"""
@@@@@@@@@@@@@@@@
@@ SEARCH TOOLS @@
@@@@@@@@@@@@@@@@
"""

class Arxiv(Command):
    def __init__(self):
        super().__init__()
        self.arxiv_eng = ArxivSearch()
        self.num_papers_per_search = 10
        self.cmd_type = "SEARCH-arxiv"

    def docstring(self) -> str:
        return (
            "============= ARXIV SEARCH TOOL ============="
            "You also have access to machine learning paper from Arxiv. "
            "To search for summaries of papers on arxiv you can use the following command: ```SUMMARY\n<search query>\n```\n where <search query> is a string that will be used as the search query to find papers with semantically similar content and SUMMARY is just the word SUMMARY.\n"
            "To get the full paper text for an arXiv paper, use the following command: ```FULL_TEXT\n<arxiv paper id>\n```\n where <arxiv paper id> is the ID of the arXiv paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
            "When you read arxiv paper, make sure to take note of the techniques they are using to solve their problem as well as the hyperparameters and implementation details. These are very important for successfully solving machine learning problems."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> command
        # args[1] -> query
        if args[0] == "SUMMARY":
            return self.arxiv_eng.find_papers_by_str(args[1], self.num_papers_per_search)
        elif args[0] == "FULL_TEXT":
            return self.arxiv_eng.retrieve_full_paper_text(args[1])
        raise Exception("Invalid Arxiv Search")

    def matches_command(self, cmd_str) -> bool:
        if "```SUMMARY" in cmd_str: return True
        elif "```FULL_TEXT" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        sum_text = extract_prompt(args[0], "SUMMARY").split("\n")
        full_text = extract_prompt(args[0], "FULL_TEXT").split("\n")
        if len(sum_text) == 0 and len(full_text) == 0: return False, None
        if len(sum_text) > 0: return True, ("SUMMARY", sum_text,)
        if len(full_text) > 0: return True, ("FULL_TEXT", full_text,)

"""
@@@@@@@@@@@@@@@@@@@
@@ WRITING TOOLS @@
@@@@@@@@@@@@@@@@@@@
"""

class PaperReplace(Command):
    def __init__(self, save_loc):
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-replace"

    def docstring(self) -> str:
        return (
            "============= PAPER REPLACING TOOL =============\n"
            "You also have access to a paper replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current latex and erase all existing latex.\n"
            "You can use this tool via the following command: ```REPLACE\n<latex here>\n```, where REPLACE is the word REPLACE and <latex here> will be the newlatex that is replacing the entire set of old latex. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Try limiting the use of rewriting and aim for editing the latex more."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new latex
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        new_latex = extract_prompt(args[0], "REPLACE")
        latex_ret = compile_latex(new_latex, self.save_loc, compile=args[1])
        if "[CODE EXECUTION ERROR]" in latex_ret: return False, (None, latex_ret,)
        return True, (new_latex.split("\n"), latex_ret)

class PaperEdit(Command):
    def __init__(self, save_loc):
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-edit"

    def docstring(self) -> str:
        return (
            "============= PAPER EDITING TOOL =============\n"
            "You also have access to a paper editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current latex with as many lines of new latex as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with latex. \n"
            "You can edit latex using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new latex that is replacing the old latex. Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Your changes should significantly change the latex. You should write new paragraphs and update old ones. Try using the edit command often. Make sure to generate lots of text. You should also avoid editing lines 0 0, and should edit the main text of the paragraphs, such as editing lines in the middle of the text body."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> N (int)
        # args[1] -> M (int)
        # args[2] -> old latex
        # args[3] -> new lines to replace
        try:
            args = args[0]
            current_latex = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))
            for _ln in lines_to_replace:
                current_latex.pop(_ln)
            for _line in lines_to_add:
                current_latex.insert(args[0], _line)
            new_latex = "\n".join(current_latex)
            latex_exec = f"{new_latex}"
            latex_ret = compile_latex(latex_exec, self.save_loc, compile=args[4])
            if "error" in latex_ret.lower(): return (False, None, latex_ret)
            return (True, current_latex, latex_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        cmd_str, latexlines = args[0], args[1]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0: return False, (None, None, None, None)
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2: return False, (None, None, None, None)
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0: return False, (None, None, None, None)
            return success, (lines_to_edit[0], lines_to_edit[1], latexlines, text[1:])
        except Exception as e:
            return False, (None, None, None, None)

