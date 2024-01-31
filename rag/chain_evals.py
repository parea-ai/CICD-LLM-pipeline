from typing import Optional

import os

from dotenv import load_dotenv
from parea import Parea
from parea.evals.rag import percent_target_supported_by_context_factory
from parea.evals.utils import EvalFuncTuple, run_evals_in_thread_and_log
from parea.schemas.log import LLMInputs, Log
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

EVALS = [
    EvalFuncTuple(
        name="supported_by_context",
        func=percent_target_supported_by_context_factory(context_fields=["context"]),
    ),
]


def run_chain(
    chain,
    question: str,
    target: Optional[str] = None,
    run_eval: bool = True,
    verbose: bool = False,
) -> str:
    """
    Run the chain with the question and target answer and run evals in background thread.
    :param chain:
    :param question: question to ask
    :param target: target answer
    :param run_eval: whether to run evals
    :param verbose: whether to print verbose logs

    :return: str
    """
    # Init Tracer which will send logs to Parea AI
    parea_tracer = PareaAILangchainTracer()
    response = chain.get_chain().invoke({"question": question}, config={"callbacks": [parea_tracer]})
    trace_id = parea_tracer.get_parent_trace_id()
    if run_eval:
        log = Log(
            configuration=LLMInputs(model="gpt-3.5-turbo-16k"),
            inputs={"question": question, "context": chain.get_context()},
            output=response,
            target=target,
        )
        run_evals_in_thread_and_log(trace_id=str(trace_id), log=log, eval_funcs=EVALS, verbose=verbose)

    return response
