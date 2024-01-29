# pylint: disable=redefined-outer-name,missing-module-docstring,missing-function-docstring

# Third Party libs
import pytest

# Custom libs
import chains

# Define test set with example questions and expected outputs for evals
EXAMPLES = [
    (
        "what is langchain?",
        "langchain is an open-source framework for building applications using large language models. it is also the name of the company building langsmith.",
    ),
    (
        "how might i query for all runs in a project?",
        "client.list_runs(project_name='my-project-name'), or in typescript, client.listruns({projectname: 'my-project-anme'})",
    ),
    (
        "what's a langsmith dataset?",
        "a langsmith dataset is a collection of examples. each example contains inputs and optional expected outputs or references for that data point.",
    ),
    (
        "how do i use a traceable decorator?",
        """the traceable decorator is available in the langsmith python sdk. to use, configure your environment with your api key, import the required function, decorate your function, and then call the function. below is an example:
        ```python
        from langsmith.run_helpers import traceable
        @traceablelan(run_type="chain") # or "llm", etc.
        def my_function(input_param):
            # function logic goes here
            return output
        result = my_function(input_param)```""",
    ),
    (
        "can i trace my llama v2 llm?",
        "so long as you are using one of langchain's llm implementations, all your calls can be traced",
    ),
    (
        "why do i have to set environment variables?",
        "environment variables can tell your langchain application to perform tracing and contain the information necessary to authenticate to langsmith."
        " while there are other ways to connect, environment variables tend to be the simplest way to configure your application.",
    ),
    (
        "how do i move my project between organizations?",
        "langsmith doesn't directly support moving projects between organizations.",
    ),
]


# =================== SETUP =================== #

@pytest.fixture
def retriever():
    return chains.DocumentRetriever().get_retriever()


@pytest.fixture
def chain_1():
    return chains.AssistantChain()


@pytest.fixture
def chain_2(retriever):
    return chains.DocumentationChain(retriever)


# ============================================= #


# =================== TESTS =================== #


def test_name(chain_1):
    print("\n\n==== test: test_name ====")

    # Define input/output
    target = "bob"
    question = "What is your name?"
    output_text = chains.run_chain(chain_1, question, target, False)
    print("Question: " + question)
    print("Answer:   " + output_text)

    assert target in output_text.lower()


def test_basic_arithmetic(chain_1):
    print("\n\n==== test: test_basic_arithmetic ====")

    # Define input/output
    target = "12"
    question = "What is 5 + 7?"
    output_text = chains.run_chain(chain_1, question, target, False)
    print("Question: " + question)
    print("Answer:   " + output_text)

    assert target in output_text.lower()


def test_llm_evaluators_experiment(chain_2):
    print("\n\n==== test: test_llm_evaluators ====")
    e = chains.p.experiment(
        name="test_llm_evaluators_experiment",
        data=[
            {"chain": chain_2, "question": question, "target": target}
            for question, target in EXAMPLES
        ],
        func=chains.run_chain
    )
    e.run()
    avg = e.experiment_stats.cumulative_avg_score()
    print("Average Cumulative score: " + f"{avg:.2f}")
    assert avg > 0.5
