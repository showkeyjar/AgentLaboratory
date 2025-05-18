import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .main import (
    LaboratoryWorkflow,
    AgentRxiv,
    parse_arguments,
    parse_yaml,
    GLOBAL_AGENTRXIV,
    RESEARCH_DIR_PATH,
)
from utils import remove_figures, remove_directory


def main():
    user_args = parse_arguments()
    yaml_to_use = user_args.yaml_location
    args = parse_yaml(yaml_to_use)

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true" if isinstance(args.copilot_mode, str) else args.copilot_mode
    compile_pdf = args.compile_latex.lower() == "true" if isinstance(args.compile_latex, str) else args.compile_latex
    load_previous = args.load_previous.lower() == "true" if isinstance(args.load_previous, str) else args.load_previous
    parallel_labs = args.parallel_labs.lower() == "true" if isinstance(args.parallel_labs, str) else args.parallel_labs
    except_if_fail = args.except_if_fail.lower() == "true" if isinstance(args.except_if_fail, str) else args.except_if_fail
    agentRxiv = args.agentRxiv.lower() == "true" if isinstance(args.agentRxiv, str) else args.agentRxiv
    construct_agentRxiv = args.construct_agentRxiv.lower() == "true" if isinstance(args.construct_agentRxiv, str) else args.construct_agentRxiv
    lab_index = int(args.lab_index) if isinstance(args.construct_agentRxiv, str) else args.lab_index

    try:
        num_papers_to_write = int(args.num_papers_to_write.lower()) if isinstance(args.num_papers_to_write, str) else args.num_papers_to_write
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower()) if isinstance(args.num_papers_lit_review, str) else args.num_papers_lit_review
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower()) if isinstance(args.papersolver_max_steps, str) else args.papersolver_max_steps
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower()) if isinstance(args.mlesolver_max_steps, str) else args.mlesolver_max_steps
    except Exception:
        raise Exception("args.mlesolver_max_steps must be a valid integer!")

    if parallel_labs:
        num_parallel_labs = int(args.num_parallel_labs)
        print("=" * 20, f"RUNNING {num_parallel_labs} LABS IN PARALLEL", "=" * 20)
    else:
        num_parallel_labs = 0

    api_key = (os.getenv("OPENAI_API_KEY") or getattr(args, "api_key", None)) if (
        hasattr(args, "api_key") or os.getenv("OPENAI_API_KEY")
    ) else None
    deepseek_api_key = (
        os.getenv("DEEPSEEK_API_KEY") or getattr(args, "deepseek_api_key", None)
    ) if (hasattr(args, "deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")) else None
    if api_key is not None and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if deepseek_api_key is not None and os.getenv("DEEPSEEK_API_KEY") is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError(
            "API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable."
        )

    if human_mode or args.research_topic is None:
        research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else:
        research_topic = args.research_topic

    task_notes_LLM = []
    task_notes = args.task_notes
    for _task in task_notes:
        for _note in task_notes[_task]:
            task_notes_LLM.append({"phases": [_task.replace("-", " ")], "note": _note})

    if args.language != "English":
        task_notes_LLM.append(
            {
                "phases": [
                    "literature review",
                    "plan formulation",
                    "data preparation",
                    "running experiments",
                    "results interpretation",
                    "report writing",
                    "report refinement",
                ],
                "note": f"You should always write in the following language to converse and to write the report {args.language}",
            }
        )

    human_in_loop = {
        "literature review": human_mode,
        "plan formulation": human_mode,
        "data preparation": human_mode,
        "running experiments": human_mode,
        "results interpretation": human_mode,
        "report writing": human_mode,
        "report refinement": human_mode,
    }

    agent_models = {
        "literature review": llm_backend,
        "plan formulation": llm_backend,
        "data preparation": llm_backend,
        "running experiments": llm_backend,
        "report writing": llm_backend,
        "results interpretation": llm_backend,
        "paper refinement": llm_backend,
    }

    if parallel_labs:
        remove_figures()
        GLOBAL_AGENTRXIV = AgentRxiv()
        remove_directory(f"{RESEARCH_DIR_PATH}")
        os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        if not compile_pdf:
            raise Exception("PDF compilation must be used with agentRxiv!")

        def run_lab(parallel_lab_index: int):
            time_str = ""
            time_now = time.time()
            for _paper_index in range(num_papers_to_write):
                lab_dir = os.path.join(RESEARCH_DIR_PATH, f"research_dir_lab{parallel_lab_index}_paper{_paper_index}")
                os.mkdir(lab_dir)
                os.mkdir(os.path.join(lab_dir, "src"))
                os.mkdir(os.path.join(lab_dir, "tex"))
                lab_instance = LaboratoryWorkflow(
                    parallelized=True,
                    research_topic=research_topic,
                    notes=task_notes_LLM,
                    agent_model_backbone=agent_models,
                    human_in_loop_flag=human_in_loop,
                    openai_api_key=api_key,
                    compile_pdf=compile_pdf,
                    num_papers_lit_review=num_papers_lit_review,
                    papersolver_max_steps=papersolver_max_steps,
                    mlesolver_max_steps=mlesolver_max_steps,
                    paper_index=_paper_index,
                    lab_index=parallel_lab_index,
                    except_if_fail=except_if_fail,
                    lab_dir=lab_dir,
                    agentRxiv=True,
                    agentrxiv_papers=args.agentrxiv_papers,
                )
                lab_instance.perform_research()
                time_str += str(time.time() - time_now) + " | "
                with open(f"agent_times_{parallel_lab_index}.txt", "w") as f:
                    f.write(time_str)
                time_now = time.time()

        with ThreadPoolExecutor(max_workers=num_parallel_labs) as executor:
            futures = [executor.submit(run_lab, lab_idx) for lab_idx in range(num_parallel_labs)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in lab: {e}")

        raise NotImplementedError("Todo: implement parallel labs")
    else:
        remove_figures()
        if agentRxiv:
            GLOBAL_AGENTRXIV = AgentRxiv(lab_index)
        if not agentRxiv:
            remove_directory(f"{RESEARCH_DIR_PATH}")
            os.mkdir(os.path.join(".", f"{RESEARCH_DIR_PATH}"))
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        time_str = ""
        time_now = time.time()
        for _paper_index in range(num_papers_to_write):
            lab_direct = f"{RESEARCH_DIR_PATH}/research_dir_{_paper_index}_lab_{lab_index}"
            os.mkdir(os.path.join(".", lab_direct))
            os.mkdir(os.path.join(f"./{lab_direct}", "src"))
            os.mkdir(os.path.join(f"./{lab_direct}", "tex"))
            lab = LaboratoryWorkflow(
                research_topic=research_topic,
                notes=task_notes_LLM,
                agent_model_backbone=agent_models,
                human_in_loop_flag=human_in_loop,
                openai_api_key=api_key,
                compile_pdf=compile_pdf,
                num_papers_lit_review=num_papers_lit_review,
                papersolver_max_steps=papersolver_max_steps,
                mlesolver_max_steps=mlesolver_max_steps,
                paper_index=_paper_index,
                except_if_fail=except_if_fail,
                agentRxiv=False,
                lab_index=lab_index,
                lab_dir=f"./{lab_direct}",
            )
            lab.perform_research()
            time_str += str(time.time() - time_now) + " | "
            with open(f"agent_times_{lab_index}.txt", "w") as f:
                f.write(time_str)
            time_now = time.time()


if __name__ == "__main__":
    main()
