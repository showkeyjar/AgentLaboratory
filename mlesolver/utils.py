from inference import query_model
from utils import extract_prompt

GLOBAL_REPAIR_ATTEMPTS = 2


def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            sys = (
                "You are a professor agent who is serving as an expert reward model that can read a research plan, research code, and code output and are able to determine how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float.\n\n"
                "You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1 representing how well the model followed the plan, built the code, and got the proper output."
            )
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research code that the model produced: \n{code}\n\n"
                    f"The following is the output from the model: {code_return}\n\n"
                ),
                temp=0.6,
            )
            performance = extract_prompt(text=scoring, word="SCORE")
            performance = float(performance)
            return performance, f"The performance of your submission is: {performance}", True
        except Exception as e:
            return None, str(e), False
    return 0, e


def code_repair(code, error, ctype, REPAIR_LLM, openai_api_key=None):
    if ctype == "replace":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            "You must wrap the code in the following ```python\n<code here>\n```\n"
            "Do not forget the opening ```python and the closing ```."
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}",
            temp=0.8,
        )
        return extract_prompt(model_resp, "python")
    elif ctype == "edit":
        repair_sys = (
            "You are an automated code repair tool.\n"
            "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
            "Your output should match the original code as closely as possible.\n"
            "============= CODE EDITING TOOL =============\n"
            "You have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code.\n"
            "Please use the code editing tool to fix this code."
            "Do not forget the opening ```EDIT N M and the closing ```."
            "Your output should look like the following\n\n```EDIT N M\n<new lines to replace old lines>\n```"
        )
        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}",
            temp=0.2,
        )
        return model_resp
