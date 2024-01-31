# Std libs
from typing import Optional

import os
from datetime import datetime
from operator import itemgetter

# Third Party libs
from dotenv import load_dotenv

# LangChain libs
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

# Parea libs
from parea import Parea
from parea.evals.rag import percent_target_supported_by_context_factory
from parea.evals.utils import EvalFuncTuple, run_evals_in_thread_and_log
from parea.schemas.log import LLMInputs, Log
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

# Init Tracer which will send logs to Parea AI
parea_tracer = PareaAILangchainTracer()

EVALS = [
    EvalFuncTuple(
        name="supported_by_context",
        func=percent_target_supported_by_context_factory(context_fields=["context"]),
    ),
]


class AssistantChain:
    """AssistantChain
    Based on a user's prompt, this will then prompt gpt-3.5-turbo-16k
    and return the response from the llm.
    """

    def __init__(self, name: str = "Bob"):
        template = f"You are a helpful assistant who's name is {name}."
        human_template = "{question}"

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", human_template),
            ]
        )
        self.chain = chat_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()

    def get_chain(self):
        return self.chain


class DocumentRetriever:
    def __init__(self, url: str = "https://docs.smith.langchain.com"):
        api_loader = RecursiveUrlLoader(url)
        raw_documents = api_loader.load()

        # Transformer
        doc_transformer = Html2TextTransformer()
        transformed = doc_transformer.transform_documents(raw_documents)

        # Splitter
        text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=2000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(transformed)

        # Define vector store based
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        return self.retriever


class DocumentationChain:
    def __init__(self, retriever):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful documentation Q&A assistant, trained to answer"
                    " questions from LangSmith's documentation."
                    " LangChain is a framework for building applications using large language models."
                    "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages.",
                ),
                ("system", "{context}"),
                ("human", "{question}"),
            ]
        ).partial(time=str(datetime.now()))

        model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
        response_generator = prompt | model | StrOutputParser()

        # The runnable map here routes the original inputs to a context
        # and a question dictionary to pass to the response generator
        self.chain = {
            "context": itemgetter("question") | retriever | self._format_docs,
            "question": itemgetter("question"),
        } | response_generator

    def get_context(self) -> str:
        """Helper to get the context from a retrieval chain, so we can use it for evaluation metrics."""
        return self.context

    def _format_docs(self, docs) -> str:
        context = "\n\n".join(doc.page_content for doc in docs)
        # set context as an attribute, so we can access it later
        self.context = context
        return context

    def get_chain(self):
        return self.chain


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
