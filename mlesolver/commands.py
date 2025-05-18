from abc import abstractmethod
from tools.common import execute_code
from utils import extract_prompt


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


class Replace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, *args) -> str:
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret:
            return False, (None, code_ret)
        return True, (new_code.split("\n"), code_ret)


class Edit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Your changes should significantly change the functionality of the code."
        )

    def execute_command(self, *args) -> str:
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1] + 1)))
            for _ln in lines_to_replace:
                current_code.pop(_ln)
            for _line in lines_to_add:
                current_code.insert(args[0], _line)
            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            code_ret = execute_code(code_exec)
            if "CODE EXECUTION ERROR" in code_ret:
                return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0:
                return False, None
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, None
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0:
                return False, None
            return success, (
                lines_to_edit[0],
                lines_to_edit[1],
                codelines,
                text[1:],
                datasetcode,
            )
        except Exception:
            return False, (None, None, None, None, None)
