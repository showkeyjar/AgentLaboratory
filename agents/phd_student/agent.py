from ..base import BaseAgent
from inference import query_model


class PhDStudentAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "literature review",
            "plan formulation",
            "running experiments",
            "results interpretation",
            "report writing",
            "report refinement",
        ]
        self.lit_review = []

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",)
        elif phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}"
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        elif phase == "report refinement":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}\n"
                f"Current Interpretation of results: {self.interpretation}"
            )
        elif phase == "literature review":
            return sr_str
        else:
            return ""

    def requirements_txt(self):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a requirements.txt for a github repository for all of the code."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the requirements.txt below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "literature review":
            return (
                "To collect paper summaries, use the following command: ```SUMMARY\nSEARCH QUERY\n```\n where SEARCH QUERY is a string that will be used to find papers with semantically similar content and SUMMARY is just the word SUMMARY. Make sure your search queries are very short.\n"
                "To get the full paper text for an arXiv paper, use the following command: ```FULL_TEXT\narXiv paper ID\n```\n where arXiv paper ID is the ID of the arXiv paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
                "If you believe a paper is relevant to the research project proposal, you can add it to the official review after reading using the following command: ```ADD_PAPER\narXiv_paper_ID\nPAPER_SUMMARY\n```\nwhere arXiv_paper_ID is the ID of the arXiv paper, PAPER_SUMMARY is a brief summary of the paper, and ADD_PAPER is just the word ADD_PAPER. You can only add one paper at a time. \n"
                "Make sure to use ADD_PAPER when you see a relevant paper. DO NOT use SUMMARY too many times."
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "Make sure to extensively discuss the experimental results in your summary.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD_PAPER, FULL_TEXT, SUMMARY). Do not use the word COMMAND make sure to use the actual command, e.g. your command should look exactly like this: ```ADD_PAPER\ntext\n``` (where the command could be from ADD_PAPER, FULL_TEXT, SUMMARY)\n")
        elif phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        elif phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        elif phase == "results interpretation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        #elif phase == "report writing":
        #    return (
        #        "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
        #        "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n")
        elif phase == "report refinement":
            return ""
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "literature review":
            phase_str = (
                "Your goal is to perform a literature review for the presented task and add papers to the literature review.\n"
                "You have access to arXiv and can perform two search operations: (1) finding many different paper summaries from a search query and (2) getting a single full paper text for an arXiv paper.\n"
            )
            rev_papers = "Papers in your review so far: " + " ".join([_paper["arxiv_id"] for _paper in self.lit_review])
            phase_str += rev_papers if len(self.lit_review) > 0 else ""
        elif phase == "plan formulation":
            phase_str = (
                "You are a PhD student being directed by a postdoc who will help you come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment. Your idea should be very innovative and unlike anything seen before.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are a PhD student being directed by a postdoc who will help you come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the postdoc your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
            )
        #elif phase == "report writing":
        #    phase_str = (
        #        "You are a PhD student being directed by a professor who will help you write a report based on results from an experiment, and you interact with them through dialogue.\n"
        #        "Your goal is to write a report for an experiment entirely in latex. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the professor how you can write up the results and receive their feedback to improve your thoughts.\n"
        #        "Your report should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance  in latex. You must propagate this information accurately.\n"
        #        "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
        #    )
        elif phase == "report refinement":
            phase_str = (
                "You are a PhD student who has submitted their paper to an ML conference called ICLR. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the conference.\n"
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "a computer science PhD student at a top university."

    def add_review(self, review, arx_eng, agentrxiv=False, GLOBAL_AGENTRXIV=None):
        try:
            if agentrxiv:
                arxiv_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                full_text = GLOBAL_AGENTRXIV.retrieve_full_text(arxiv_id,)
            else:
                arxiv_id, review_text = review.strip().split("\n", 1)
                full_text = arx_eng.retrieve_full_paper_text(arxiv_id)
            review_entry = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            }
            self.lit_review.append(review_entry)
            return f"Successfully added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}. Your provided Arxiv ID might not be valid. Make sure it references a real paper, which can be found using the SUMMARY command.", ""

    def format_review(self):
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)



