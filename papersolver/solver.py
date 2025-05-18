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
import os, sys

from .commands import Arxiv, PaperReplace, PaperEdit
from .constants import per_section_tips

class PaperSolver:
    def __init__(self, llm_str, notes=None, max_steps=10, insights=None, plan=None, exp_code=None, exp_results=None, lit_review=None, ref_papers=None, topic=None, openai_api_key=None, compile_pdf=True, save_loc=None):
        self.supress_print = True
        if notes is None: self.notes = []
        else: self.notes = notes
        if plan is None: self.plan = ""
        else: self.plan = plan
        if exp_code is None: self.exp_code = ""
        else: self.exp_code = exp_code
        if exp_results is None: self.exp_results = ""
        else: self.exp_results = exp_results
        if lit_review is None: self.lit_review = ""
        else: self.lit_review = lit_review
        if insights is None: self.insights = ""
        else: self.insights = insights
        if ref_papers is None: self.ref_papers = ""
        else: self.ref_papers = ref_papers
        if topic is None: self.topic = ""
        else: self.topic = topic
        self.save_loc = save_loc
        self.compile_pdf = compile_pdf
        self.llm_str = llm_str
        self.notes = notes
        self.max_papers = 1
        self.st_hist_len = 10
        self.min_gen_trials = 2
        self.max_steps = max_steps
        self.paper_lines = str()
        self.prev_paper_ret = str()
        self.section_related_work = {}
        self.openai_api_key = openai_api_key

    def solve(self):
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_paper_ret = None
        while True:
            self.paper_lines = copy(random.choice(self.best_report)[0])
            model_resp = query_model(
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"\nNow please enter a command: ",
                temp=1.0,
                openai_api_key=self.openai_api_key)
            model_resp = self.clean_text(model_resp)
            cmd_str, paper_lines, prev_paper_ret, score = self.process_command(model_resp)
            if score is not None:
                if top_score is None:
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score
            if num_attempts >= self.min_gen_trials and top_score is not None: break
            if not self.supress_print: print(f"@@@ Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print: print(f"$$$ Score: {score}")
            num_attempts += 1
        self.paper_lines, self.prev_paper_ret, model_resp, cmd_str = best_pkg
        # add top scoring paper that was successful to the best papers
        if top_score > self.best_report[-1][1]:
            # replace the lowest scoring one
            if len(self.best_report) >= self.max_papers:
                self.best_report.pop(-1)
            self.best_report.append((copy(self.paper_lines), copy(top_score), self.prev_paper_ret))
            # sort by score, to make sure lowest are removed in future
            self.best_report.sort(key=lambda x: x[1], reverse=True)
        return model_resp, cmd_str

    def initial_solve(self):
        """
        Initialize the solver and get an initial set of papers and a return
        @return: None
        """
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@ Initial PaperGen Commands @@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.best_score = None
        self.commands = [PaperReplace(self.save_loc)]
        self.model = f"{self.llm_str}"
        init_report, init_return, self.best_score = self.gen_initial_report()
        self.best_report = [(copy(init_report), self.best_score, init_return) for _ in range(1)]

        self.paper_lines = init_report
        self.model = f"{self.llm_str}"
        self.commands = [PaperEdit(self.save_loc)] #, Replace()]
        self.prev_working_report = copy(self.paper_lines)

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")
        return text

    def gen_initial_report(self):
        num_attempts = 0
        arx = ArxivSearch()
        section_scaffold = str()
        #  1. Abstract 2. Introduction, 3. Background, 4. Methods, 5. Experimental Setup 6. Results, and 7. Discussion
        for _section in ["scaffold", "abstract", "introduction", "related work", "background", "methods", "experimental setup", "results", "discussion"]:
            section_complete = False
            if _section in ["introduction", "related work", "background", "methods", "discussion"]:
                attempts = 0
                papers = str()
                first_attempt = True
                while len(papers) == 0:
                    att_str = str()
                    if attempts > 5:
                        break
                    if not first_attempt:
                        att_str = "This is not your first attempt please try to come up with a simpler search query."
                    search_query = query_model(model_str=f"{self.llm_str}", prompt=f"Given the following research topic {self.topic} and research plan: \n\n{self.plan}\n\nPlease come up with a search query to find relevant papers on arXiv. Respond only with the search query and nothing else. This should be a a string that will be used to find papers with semantically similar content. {att_str}", system_prompt=f"You are a research paper finder. You must find papers for the section {_section}. Query must be text nothing else.", openai_api_key=self.openai_api_key)
                    search_query.replace('"', '')
                    papers = arx.find_papers_by_str(query=search_query, N=10)
                    first_attempt = False
                    attempts += 1
                if len(papers) != 0:
                    self.section_related_work[_section] = papers
            while not section_complete:
                section_scaffold_temp = copy(section_scaffold)
                if num_attempts == 0: err = str()
                else: err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the presented problem."
                if _section == "scaffold":
                    prompt = f"{err}\nNow please enter the ```REPLACE command to create the scaffold:\n "
                else:
                    rp = str()
                    if _section in self.section_related_work:
                        rp = f"Here are related papers you can cite: {self.section_related_work[_section]}. You can cite them just by putting the arxiv ID in parentheses, e.g. (arXiv 2308.11483v1)\n"
                    prompt = f"{err}\n{rp}\nNow please enter the ```REPLACE command to create the designated section, make sure to only write the text for that section and nothing else. Do not include packages or section titles, just the section content:\n "
                model_resp = query_model(
                    model_str=self.model,
                    system_prompt=self.system_prompt(section=_section),
                    prompt=f"{prompt}",
                    temp=0.8,
                    openai_api_key=self.openai_api_key)
                model_resp = self.clean_text(model_resp)
                if _section == "scaffold":
                    # minimal scaffold (some other sections can be combined)
                    for _sect in ["[ABSTRACT HERE]", "[INTRODUCTION HERE]", "[METHODS HERE]", "[RESULTS HERE]", "[DISCUSSION HERE]"]:
                        if _sect not in model_resp:
                            cmd_str = "Error: scaffold section placeholders were not present (e.g. [ABSTRACT HERE])."
                            if not self.supress_print: print("@@@ INIT ATTEMPT:", cmd_str)
                            continue
                elif _section != "scaffold":
                    new_text = extract_prompt(model_resp, "REPLACE")
                    section_scaffold_temp = section_scaffold_temp.replace(f"[{_section.upper()} HERE]", new_text)
                    model_resp = '```REPLACE\n' + copy(section_scaffold_temp) + '\n```'
                    if "documentclass{article}" in new_text or "usepackage{" in new_text:
                            cmd_str = "Error: You must not include packages or documentclass in the text! Your latex must only include the section text, equations, and tables."
                            if not self.supress_print: print("@@@ INIT ATTEMPT:", cmd_str)
                            continue
                cmd_str, latex_lines, prev_latex_ret, score = self.process_command(model_resp, scoring=False)
                if not self.supress_print: print(f"@@@ INIT ATTEMPT: Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
                #print(f"$$$ Score: {score}")
                if score is not None:
                    section_complete = True
                    section_scaffold = "\n".join(latex_lines)
                num_attempts += 1
            self.paper_lines = section_scaffold.split("\n")
            if not self.supress_print: print("$"*10, f"SCAFFOLD [{_section}] CREATED", "$"*10)
        if not self.supress_print: print("$"*10, "SCAFFOLD CREATED", "$"*10)
        return latex_lines, prev_latex_ret, score

    def process_command(self, model_resp, scoring=True):
        """
        Take command from language model and execute if valid
        @param model_resp: (str) language model output
        @return: (tuple) tuple containing the following items
            - cmd_str: (str) paper execution return and success flag
            - paper_lines: (list) list of paper lines as strings
            - prev_paper_ret: (str) output from running paper
            - score: (float) score of model
        """
        cmd_str = None
        score = None
        prev_paper_ret = self.prev_paper_ret
        paper_lines = copy(self.paper_lines)
        if "\\includegraphics[width=\\textwidth]{Figure_1.png}" in model_resp or "\\includegraphics[width=\\textwidth]{Figure_2.png}" in model_resp:
            cwd = os.getcwd()
            model_resp = model_resp.replace("\\includegraphics[width=\\textwidth]{Figure_1.png}", "\\includegraphics[width=\\textwidth]{" + cwd + "/Figure_1.png}")
            model_resp = model_resp.replace("\\includegraphics[width=\\textwidth]{Figure_2.png}", "\\includegraphics[width=\\textwidth]{" + cwd + "/Figure_2.png}")
        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                # attempt to execute the paper edit command
                if cmd.cmd_type == "PAPER-edit": # DONE
                    score = None
                    failed = True
                    success, args = cmd.parse_command(model_resp, paper_lines)
                    paper_err = f"Return from executing latex: {args[1]}"
                    if success:
                        # True, current_latex, latex_ret
                        args = cmd.execute_command((args[0], args[1], paper_lines, args[3], self.compile_pdf))
                        success = success and args[0]
                        if not success: pass
                        else:
                            paper_lines = copy(args[1]) #
                            if scoring:
                                score, cmd_str, is_valid = get_score(self.plan, "\n".join(paper_lines), reward_model_llm=self.llm_str)
                            else:
                                score, cmd_str, is_valid = 0.0, "Paper scored successfully", True
                            if is_valid: failed = False
                            paper_err += f"\nReturn from executing latex: {cmd_str}"
                        if not self.supress_print: print("$$$$ PAPER EDIT (success)")
                    if failed:
                        cmd_str = f"Paper edit FAILED due to the following error: {paper_err}.  Paper was reverted back to original state before edits."
                        if not self.supress_print: print("$$$$ PAPER EDIT (failed)")
                    else:
                        cmd_str = "Paper was successfully edited."
                        paper_lines = copy(args[1])
                        prev_paper_ret = copy(args[2])
                        if not self.supress_print: print("$$$$ PAPER EDIT (success)")
                elif cmd.cmd_type == "PAPER-replace": # DONE
                    score = None
                    failed = True
                    success, args = cmd.parse_command(model_resp, self.compile_pdf)
                    paper_err = f"Return from executing latex: {args[1]}"
                    if success:
                        paper_lines = copy(args[0]) #
                        if scoring:
                            score, cmd_str, is_valid = get_score(self.plan, "\n".join(paper_lines), reward_model_llm=self.llm_str)
                        else:
                            score, cmd_str, is_valid = 0.0, "Paper scored successfully", True
                        if is_valid: failed = False
                        paper_err += f"\nReturn from executing code on real test set {cmd_str}"
                    if failed:
                        cmd_str = f"Paper replacement FAILED due to the following error: {paper_err}.  Paper was reverted back to original state before edits."
                        if not self.supress_print: print("$$$$ PAPER REPLACE (failed)")
                    else:
                        cmd_str = "Paper was successfully replaced."
                        paper_lines = copy(args[0])
                        prev_paper_ret = copy(args[1])
                        if not self.supress_print: print("$$$$ PAPER REPLACE (success)")
        return cmd_str, paper_lines, prev_paper_ret, score

    def generate_paper_lines(self, code):
        """
        Generate well-formatted code lines with line numbers
        @param code: (list) list of code line strings
        @return: (str) code lines formatted with line numbers
        """
        codestr = str()
        for _index in range(len(code)):
            codestr += f"{_index} |{code[_index]}\n"
        return codestr

    def system_prompt(self, commands=True, section=None):
        """
        Produce a system prompt for the paper-solver
        @param commands: (bool) whether to use command prompt
        @return: (str) system prompt
        """
        if section == "abstract": length = "This section should be ONLY 1 paragraph."
        else: length = "This section should be approximately 2-4 paragraphs and so your output should be several paragraphs of latex."
        methods_str = str()
        if section == "methods":
            fig1_text="""\n\\begin{figure}[h]
\\caption{<caption here>}
\\centering
\\includegraphics[width=\\textwidth]{Figure_1.png}
\\label{fig:fig1}
\\end{figure}\n"""
            fig2_text="""\n\\begin{figure}[h]
\\caption{<caption here>}
\\centering
\\includegraphics[width=\\textwidth]{Figure_2.png}
\\label{fig:fig1}
\\end{figure}\n"""
            if os.path.exists("Figure_1.png") and os.path.exists("Figure_2.png"):
                methods_str += f"You ABSOLUTELY must without fail also include Figure_1.png and Figure_2.png in your paper using {fig1_text} and {fig2_text} on a new line. Make sure to place these figures in separate locations."
            elif os.path.exists("Figure_1.png"):
                methods_str += f"You ABSOLUTELY must without fail also include Figure_1.png in your paper using {fig1_text} on a new line.\n"
            elif os.path.exists("Figure_2.png"):
                methods_str += f"You ABSOLUTELY must without fail also include Figure_2.png in your paper using {fig2_text} on a new line.\n"
        if section is not None and section == "scaffold": section_cmd = f"Your objective right now is to only build the scaffolding for the paper. You should not include any text in the body of the paper, but should have an empty scaffold for each of the sections.  Where the sections go, write [ABSTRACT HERE] for abstract, and write [INTRODUCTION HERE] for the introduction... etc. Your paper should have the following sections: 1. Abstract 2. Introduction, 3. Background, 4. Related Work 5. Methods, 6. Experimental Setup 7. Results, and 8. Discussion. Just create the scaffolding as compilable latex. Your title should start with Research Report: [title here] where title here is a title you choose. For author write Agent Laboratory."
        elif section is not None: section_cmd = f"Your only goal is to generate latex for the following {section}. DO NOT INCLUDE ANY PACKAGES OR ANY SECTION COMMANDS. DO NOT INCLUDE A TITLE OR DATE ONLY TEXT. You only have to generate text for this specific section and do not have to output anything else. {length} I repeat DO NOT INCLUDE ANY PACKAGES OR ANY SECTION COMMANDS. DO NOT INCLUDE A TITLE OR DATE ONLY TEXT. Use as many equations as you find necessary. You should include mathematical equations, numbers, and tables where necessary. Remember that to include a percentage sign % you must add a backslash \% or else it will become a comment. Here are some tips {per_section_tips[section]}  {methods_str}.\n\n"
        else: section_cmd = ""
        paper_len = sum([i.strip(string.punctuation).isalpha() for i in ("".join(self.paper_lines)).split()])
        #paper_len2 = len(("".join(self.paper_lines)).split())
        if paper_len < 4000: paper_progress = f"The current length of the paper is {paper_len} words, you must increase this by {4000-paper_len} words."
        else: paper_progress = ""
        if not self.supress_print: print(paper_progress)
        cmd_set = f"The following are commands you have access to: {self.command_descriptions()}\n." if commands else ""
        if len(self.ref_papers) == 0: ref_papers = ""
        else:
            refpapers = '\n'.join(self.ref_papers)
            ref_papers = f"Here is a reference paper that is high quality:\n{refpapers}\n\n\n"
        lit_review_str = str(self.lit_review)[:20000]
        return (
            f"{ref_papers}"
            # ROLE DESCRIPTION
            f"{self.role_description()}.\n"
            # TASK INSTRUCTIONS
            f"The following are your task instructions: {self.phase_prompt()}\n"
            # NOTES
            f"The following are notes, instructions, and general tips for you: {self.notes}"
            # LIT REVIEW
            f"The following literature review was provided for the paper:\n{lit_review_str}\n"
            # PLAN DESCRIPTION
            f"You are given a paper report writing task. The original research plan was described as follows: {self.plan}\n"
            # EXPERIMENT CODE
            f"A team of research wrote the following code, following this plan: {self.exp_code}\n"
            # EXPERIMENT RESULTS
            f"After running this code, the following results were observed: {self.exp_results}\n Your results must ACCURATELY reflect the numbers presented here."
            # EXPERIMENT RESULT INSIGHTS
            f"Provided was an interpretation of the experimental results:\n{self.insights}\n"
            f"Your writing style should be boring and objective.\n"
            # transition
            f"Your goal is to write a research paper as well as possible. You will receive a score after you write the paper and should aim to maximize the score by writing a high quality research paper. The paper length should be 8 pages or 4000 words in total. It should be quite long and comprehensive. Remember, the paper MUST BE LONG. {paper_progress}\n"
            # COMMAND SET
            f"{cmd_set}\n"
            # PAPER
            f"Provided here is your current paper {self.generate_paper_lines(self.paper_lines)}"
            # optional section command
            f"{section_cmd}"
        )

    def command_descriptions(self):
        """
        Provide command descriptions
        @return: (str) command descriptions
        """
        cmd_strings = "\n".join([_cmd.docstring() for _cmd in self.commands])
        return f"\nYou also have access to tools which can be interacted with using the following structure: ```COMMAND\n<command information here>\n```, where COMMAND is whichever command you want to run (e.g. EDIT,...), <command information here> is information used for the command and ``` are meant to encapsulate the command. ``` must be included as part of the command both at the beginning and at the end of the command. DO NOT FORGOT TO HAVE ``` AT THE TOP AND BOTTOM OF COMMAND. and this structure must be followed to execute a command correctly. YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME! Do not try to perform multiple commands EVER only one." + cmd_strings

    def role_description(self):
        """
        Provide role description
        @return: (str) role description
        """
        return "You are a computer science PhD student at a top university who has submitted their paper to an ML conference called ICLR. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the conference. Your paper should be approximately 8 pages and around 4000 words. Your article should ONLY CONTAIN EIGHT sections as follows: 1. Abstract 2. Introduction, 3. Background, 4. Related Work 5. Methods, 6. Experimental Setup 7. Results, and 8. Discussion.\n"


    def phase_prompt(self,):
        """
        Describe system role and general tips for mle-solver
        @return: (str) system role
        """
        phase_str = (
            "You are a PhD student who has submitted a paper to an ML conference called ICLR. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the conference.\n"
        )
        return phase_str



