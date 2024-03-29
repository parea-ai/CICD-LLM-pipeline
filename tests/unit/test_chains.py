# pylint: disable=redefined-outer-name,missing-module-docstring,missing-function-docstring
import pytest

from rag import chains


# =================== SETUP =================== #


@pytest.fixture
def chain_1():
    return chains.AssistantChain()


# ============================================= #


# =================== TESTS =================== #


def test_name(chain_1):
    print("\n\n==== test: test_name ====")

    # Define input/output
    target = "bob"
    question = "What is your name?"
    output_text = chain_1.get_chain().invoke({"question": question})
    print("Question: " + question)
    print("Answer:   " + output_text)

    assert target in output_text.lower()


def test_basic_arithmetic(chain_1):
    print("\n\n==== test: test_basic_arithmetic ====")

    # Define input/output
    target = "12"
    question = "What is 5 + 7?"
    output_text = chain_1.get_chain().invoke({"question": question})
    print("Question: " + question)
    print("Answer:   " + output_text)

    assert target in output_text.lower()
