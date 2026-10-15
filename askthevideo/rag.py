import groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from askthevideo.config import EMBEDDING_MODEL, Settings
from askthevideo.formatting import format_context

SYSTEM_PROMPT = """You are an assistant that answers questions about one YouTube video.
Use ONLY the transcript excerpts in the context below.

Rules:
1. If the answer is not in the context, reply: "This isn't covered in the video."
2. Cite the timestamp of every excerpt you use, exactly as shown in the context, like [04:32].
3. Keep answers short and clear. Use bullet points for multi-part answers.
4. The transcript is auto-generated and may contain errors. Never invent facts to fill gaps.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


class AnswerError(Exception):
    pass


def load_embeddings() -> HuggingFaceEmbeddings:
    # Unit length vectors make FAISS L2 distance rank results the same as cosine similarity.
    # The e5 models were trained with these prefixes and retrieve worse without them.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
    )


def create_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def build_vector_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings)


def build_retriever(vector_store: VectorStore) -> VectorStoreRetriever:
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}
    )


def build_chain(retriever: VectorStoreRetriever, llm: BaseChatModel) -> Runnable:
    return (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(chain: Runnable, question: str) -> str:
    question = question.strip()
    if not question:
        raise AnswerError("Type a question first.")
    try:
        return chain.invoke(question)
    except groq.RateLimitError as exc:
        raise AnswerError("Groq rate limit reached. Wait a minute and try again.") from exc
    except groq.AuthenticationError as exc:
        raise AnswerError(
            "Groq rejected the API key. Check GROQ_API_KEY in your .env file and restart the app."
        ) from exc
    except groq.APIConnectionError as exc:
        raise AnswerError(
            "Could not reach the Groq API. Check your internet connection and try again."
        ) from exc
    except groq.APIStatusError as exc:
        raise AnswerError(
            f"The Groq API returned an error (HTTP {exc.status_code}). Try again shortly."
        ) from exc
    except groq.GroqError as exc:
        raise AnswerError("The Groq API request failed. Try again shortly.") from exc
