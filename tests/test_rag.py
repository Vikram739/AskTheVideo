import groq
import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from askthevideo.rag import (
    SYSTEM_PROMPT,
    AnswerError,
    ask,
    build_chain,
    build_retriever,
    build_vector_store,
)

DOCS = [
    Document(
        page_content="Gradient descent updates weights.",
        metadata={"video_id": "vid", "start": 3725.0, "end": 3785.0},
    ),
    Document(
        page_content="Welcome to the lecture on neural networks.",
        metadata={"video_id": "vid", "start": 0.0, "end": 60.0},
    ),
    Document(
        page_content="Backpropagation computes gradients.",
        metadata={"video_id": "vid", "start": 272.0, "end": 332.0},
    ),
]


class MessageRecorder(BaseCallbackHandler):
    def __init__(self):
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages.extend(messages)


@pytest.fixture
def retriever():
    vector_store = build_vector_store(DOCS, DeterministicFakeEmbedding(size=32))
    return build_retriever(vector_store)


def test_retriever_uses_mmr(retriever):
    assert retriever.search_type == "mmr"
    assert retriever.search_kwargs == {"k": 4, "fetch_k": 20}


def test_chain_sends_formatted_context_to_model(retriever):
    llm = FakeListChatModel(responses=["Backpropagation computes gradients [04:32]."])
    recorder = MessageRecorder()
    chain = build_chain(retriever, llm)

    answer = chain.invoke("What is backpropagation?", config={"callbacks": [recorder]})

    assert answer == "Backpropagation computes gradients [04:32]."
    [messages] = recorder.messages
    system, human = messages
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    expected_context = (
        "[00:00] Welcome to the lecture on neural networks.\n\n"
        "[04:32] Backpropagation computes gradients.\n\n"
        "[1:02:05] Gradient descent updates weights."
    )
    assert system.content == SYSTEM_PROMPT.format(context=expected_context)
    assert human.content == "What is backpropagation?"


def test_system_prompt_rules_text():
    assert 'reply: "This isn\'t covered in the video."' in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith("Context:\n{context}")


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_ask_rejects_empty_question(question):
    chain = RunnableLambda(lambda q: pytest.fail("chain should not run"))
    with pytest.raises(AnswerError, match="Type a question"):
        ask(chain, question)


def test_ask_strips_question():
    chain = RunnableLambda(lambda q: f"got {q!r}")
    assert ask(chain, "  why?  ") == "got 'why?'"


REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def status_error(cls, status):
    return cls("error", response=httpx.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            status_error(groq.RateLimitError, 429),
            "Groq rate limit reached. Wait a minute and try again.",
        ),
        (status_error(groq.AuthenticationError, 401), "Groq rejected the API key"),
        (status_error(groq.InternalServerError, 503), r"HTTP 503"),
        (groq.APIConnectionError(request=REQUEST), "Could not reach the Groq API"),
    ],
)
def test_ask_maps_groq_errors(error, expected):
    def fail(question):
        raise error

    with pytest.raises(AnswerError, match=expected):
        ask(RunnableLambda(fail), "question")
