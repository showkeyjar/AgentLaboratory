import PyPDF2
import threading
from app import *
from agents.professor import ProfessorAgent
from agents.postdoc import PostdocAgent
from agents.phd_student import PhDStudentAgent
from agents.ml_engineer import MLEngineerAgent
from agents.sw_engineer import SWEngineerAgent
from agents.reviewers import ReviewersAgent
from tools.common import HFDataSearch, ArxivSearch, execute_code
from copy import copy
from pathlib import Path
from datetime import date
from common_imports import *
from mlesolver import MLESolver
import argparse, pickle, yaml

GLOBAL_AGENTRXIV = None
DEFAULT_LLM_BACKBONE = "o3-mini"
RESEARCH_DIR_PATH = "MATH_research_dir"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaboratoryWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None, compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=5, paper_index=0, except_if_fail=False, parallelized=False, lab_dir=None, lab_index=0, agentRxiv=False, agentrxiv_papers=5):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """
        self.agentRxiv = agentRxiv
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        self.lab_index = lab_index
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key
        self.except_if_fail = except_if_fail
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True # should review be overridden?
        self.review_ovrd_steps = 0 # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0 # num steps to take if overridden
        self.arxiv_num_summaries = 5
        self.num_agentrxiv_papers = agentrxiv_papers
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            # todo: check if valid
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)


    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        with open(f"state_saves/Paper{self.paper_index}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.agentRxiv:
                    if self.verbose: print(f"{'&' * 30}\n[Lab #{self.lab_index} Paper #{self.paper_index}] Beginning subtask: {subtask}\n{'&' * 30}")
                else:
                    if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else: self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose: print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate mle-solver
        from papersolver import PaperSolver
        self.reference_papers = []
        solver = PaperSolver(notes=report_notes, max_steps=self.papersolver_max_steps, plan=self.phd.plan, exp_code=self.phd.results_code, exp_results=self.phd.exp_results, insights=self.phd.interpretation, lit_review=self.phd.lit_review, ref_papers=self.reference_papers, topic=research_topic, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["report writing"], compile_pdf=compile_pdf, save_loc=self.lab_dir)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps): solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        match = re.search(r'\\title\{([^}]*)\}', report)
        if match: report_title = match.group(1).replace(" ", "_")
        else: report_title = "\n".join([str(random.randint(0, 10)) for _ in range(10)])
        if self.agentRxiv: shutil.copyfile(self.lab_dir + "/tex/temp.pdf", f"uploads/{report_title}.pdf")
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file(f"./{self.lab_dir}", "readme.md", readme)
        save_to_file(f"./{self.lab_dir}", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, notes=experiment_notes, insights=self.ml_engineer.lit_review_sum, max_steps=self.mlesolver_max_steps, plan=self.ml_engineer.plan, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["running experiments"])
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        #execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry: return retry
        save_to_file(f"./{self.lab_dir}/src", "run_experiments.py", code)
        save_to_file(f"./{self.lab_dir}/src", "experiment_output.log", exp_results)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else: ml_feedback_in = ""
            resp = self.sw_engineer.inference(self.research_topic, "data preparation", feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", step=_i)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose: print("#"*40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#"*40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file(f"./{self.lab_dir}/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i)
            #if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose: print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            # inference postdoc to
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Plan Formulation")
        else:
            plan = "No plan specified."
            if self.human_in_loop_flag["plan formulation"]:
                retry = self.human_in_loop("plan formulation", plan)
                if retry: return retry
            self.set_agent_attr("plan", plan)
            # reset agent state
            self.reset_agents()
            return False

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps # lit review often requires extra steps
        # get initial response from PhD agent
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)
        if self.verbose: print(resp, "\n~~~~~~~~~~~")
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            feedback = str()
            # grab summary of papers from arxiv
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                if self.agentRxiv:
                    if GLOBAL_AGENTRXIV.num_papers() > 0:
                        papers += GLOBAL_AGENTRXIV.search_agentrxiv(query, self.num_agentrxiv_papers,)
                feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            # grab full text from arxiv ID
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                if self.agentRxiv and "AgentRxiv" in query: full_text = GLOBAL_AGENTRXIV.retrieve_full_text(query,)
                else: full_text = arx_eng.retrieve_full_paper_text(query)
                # expiration timer so that paper does not remain in context too long
                arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + full_text + "```"
                feedback = arxiv_paper

            # if add paper, extract and add to lit review, provide feedback
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                if self.agentRxiv and "AgentRxiv" in query: feedback, text = self.phd.add_review(query, arx_eng, agentrxiv=True, GLOBAL_AGENTRXIV=GLOBAL_AGENTRXIV)
                else: feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            # completion condition
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "literature review", feedback=feedback, step=_i + 1, temp=0.4)
            if self.verbose: print(resp, "\n~~~~~~~~~~~")
        if self.except_if_fail: raise Exception("Max tries during phase: Literature Review")
        else:
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False

class AgentRxiv:
    def __init__(self, lab_index=0):
        self.lab_index = lab_index
        self.server_thread = None
        self.initialize_server()
        self.pdf_text = dict()
        self.summaries = dict()

    def initialize_server(self):
        # Calculate the port dynamically
        port = 5000 + self.lab_index
        # Start the server on the computed port using a lambda to pass the port value
        self.server_thread = threading.Thread(target=lambda: self.run_server(port))
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(5)  # allow time for the server to start up

    @staticmethod
    def num_papers():
        return len(os.listdir("uploads"))

    def retrieve_full_text(self, arxiv_id):
        try:
            return self.pdf_text[arxiv_id]
        except Exception:
            return "Paper ID not found?"

    @staticmethod
    def read_pdf_pypdf2(pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def search_agentrxiv(self, search_query, num_papers):
        # Use the dynamic port here as well
        url = f'http://127.0.0.1:{5000 + self.lab_index}/api/search?q={search_query}'
        return_str = str()
        try:
            with app.app_context():
                update_papers_from_uploads()
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return_str += "Search Query:" + data['query']
            return_str += "Results:"
            for result in data['results'][:num_papers]:
                arxiv_id = f"AgentRxiv:ID_{result['id']}"
                if arxiv_id not in self.summaries:
                    filename = Path(f'_tmp_{self.lab_index}.pdf')
                    response = requests.get(result['pdf_url'])
                    filename.write_bytes(response.content)
                    self.pdf_text[arxiv_id] = self.read_pdf_pypdf2(f'_tmp_{self.lab_index}.pdf')
                    self.summaries[arxiv_id] = query_model(
                        prompt=self.pdf_text[arxiv_id],
                        system_prompt="Please provide a 5 sentence summary of this paper.",
                        openai_api_key=os.getenv('OPENAI_API_KEY'),
                        model_str="gpt-4o-mini"
                    )
                return_str += f"Title: {result['filename']}"
                return_str += f"Summary: {self.summaries[arxiv_id]}\n"
                formatted_date = date.today().strftime("%d/%m/%Y")
                return_str += f"Publication Date: {formatted_date}\n"
                return_str += f"arXiv paper ID: AgentRxiv:ID_{result['id']}"
                return_str += "-" * 40
        except Exception as e:
            print(f"AgentRxiv Error: {e}")
            return_str += f"Error: {e}"
        return return_str

    def run_server(self, port):
        run_app(port=port)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    default_yaml = Path(__file__).resolve().parent.parent / "experiment_configs" / "MATH_agentlab.yaml"

    parser.add_argument(
        '--yaml-location',
        type=str,
        default=str(default_yaml),
        help='Location of YAML to load config data.'
    )

    return parser.parse_args()


def parse_yaml(yaml_file_loc):
    yaml_path = Path(yaml_file_loc)
    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).resolve().parent.parent / yaml_path
    with open(yaml_path, 'r') as file:
        agentlab_data = yaml.safe_load(file)
    class YamlDataHolder:
        def __init__(self): pass
    parser = YamlDataHolder()
    if "copilot_mode" in agentlab_data: parser.copilot_mode = agentlab_data["copilot_mode"]
    else: parser.copilot_mode = False
    if 'load-previous' in agentlab_data: parser.load_previous = agentlab_data["load-previous"]
    else: parser.load_previous = False
    if 'research-topic' in agentlab_data: parser.research_topic = agentlab_data["research-topic"]
    if 'api-key' in agentlab_data: parser.api_key = agentlab_data["api-key"]
    if 'deepseek-api-key' in agentlab_data: parser.deepseek_api_key = agentlab_data["deepseek-api-key"]
    if 'compile-latex' in agentlab_data: parser.compile_latex = agentlab_data["compile-latex"]
    else: parser.compile_latex = True
    if 'llm-backend' in agentlab_data: parser.llm_backend = agentlab_data["llm-backend"]
    else: parser.llm_backend = "o3-mini"
    if 'lit-review-backend' in agentlab_data: parser.lit_review_backend = agentlab_data["lit-review-backend"]
    else: parser.lit_review_backend = "gpt-4o-mini"
    if 'language' in agentlab_data: parser.language = agentlab_data["language"]
    else: parser.language = "English"
    if 'num-papers-lit-review' in agentlab_data: parser.num_papers_lit_review = agentlab_data["num-papers-lit-review"]
    else: parser.num_papers_lit_review = 5
    if 'mlesolver-max-steps' in agentlab_data: parser.mlesolver_max_steps = agentlab_data["mlesolver-max-steps"]
    else: parser.mlesolver_max_steps = 3
    if 'papersolver-max-steps' in agentlab_data: parser.papersolver_max_steps = agentlab_data["papersolver-max-steps"]
    else: parser.papersolver_max_steps = 5
    if 'task-notes' in agentlab_data: parser.task_notes = agentlab_data["task-notes"]
    else: parser.task_notes = []
    if 'num-papers-to-write' in agentlab_data: parser.num_papers_to_write = agentlab_data["num-papers-to-write"]
    else: parser.num_papers_to_write = 100
    if 'parallel-labs' in agentlab_data: parser.parallel_labs = agentlab_data["parallel-labs"]
    else: parser.parallel_labs = False
    if 'num-parallel-labs' in agentlab_data: parser.num_parallel_labs = agentlab_data["num-parallel-labs"]
    else: parser.num_parallel_labs = 8
    if 'except-if-fail' in agentlab_data: parser.except_if_fail = agentlab_data["except-if-fail"]
    else: parser.except_if_fail = False
    if 'agentRxiv' in agentlab_data: parser.agentRxiv = agentlab_data["agentRxiv"]
    else: parser.agentRxiv = False
    if 'construct-agentRxiv' in agentlab_data: parser.construct_agentRxiv = agentlab_data["construct-agentRxiv"]
    else: parser.construct_agentRxiv = False
    if 'agentrxiv-papers' in agentlab_data: parser.agentrxiv_papers = agentlab_data["agentrxiv-papers"]
    else:  parser.agentrxiv_papers = 5

    if 'lab-index' in agentlab_data: parser.lab_index = agentlab_data["lab-index"]
    else: parser.lab_index = 0
    return parser


