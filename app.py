import streamlit as st

from askthevideo.chunking import chunk_transcript
from askthevideo.config import MissingApiKeyError, Settings, load_settings
from askthevideo.formatting import format_timestamp, link_timestamps
from askthevideo.rag import (
    AnswerError,
    ask,
    build_chain,
    build_retriever,
    build_vector_store,
    create_llm,
    load_embeddings,
)
from askthevideo.youtube import (
    InvalidVideoLinkError,
    TranscriptError,
    fetch_transcript,
    parse_video_id,
)

SETUP_STEPS = """
**GROQ_API_KEY is not set.** To fix it:

1. Create a free API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Copy `.env.example` to `.env` in the project folder.
3. Set `GROQ_API_KEY=<your key>` in `.env`.
4. Restart the app with `streamlit run app.py`.
"""

HOW_IT_WORKS = [
    (":material/link:", "Paste a link", "Any YouTube video with captions, in any language."),
    (":material/forum:", "Ask questions", "Answers come only from what is said in the video."),
    (":material/schedule:", "Jump to the moment", "Click a timestamp to open the video at that point."),
]

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_display:"}

SESSION_DEFAULTS = {
    "video_id": None,
    "language": None,
    "duration": None,
    "messages": [],
    "vector_store": None,
    "chain": None,
}


@st.cache_resource(show_spinner="Loading embedding model (first run only)")
def get_embeddings():
    return load_embeddings()


@st.cache_resource
def get_llm(api_key: str, model: str):
    return create_llm(Settings(groq_api_key=api_key, groq_model=model))


@st.cache_data(show_spinner=False)
def get_transcript(video_id: str) -> dict:
    return fetch_transcript(video_id)


def reset_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def load_video(link: str, settings: Settings) -> None:
    try:
        video_id = parse_video_id(link)
    except InvalidVideoLinkError as exc:
        st.error(str(exc), icon=":material/error:")
        return

    error = None
    with st.status("Loading video", expanded=True) as status:
        st.write("Fetching transcript")
        try:
            transcript = get_transcript(video_id)
        except TranscriptError as exc:
            error = str(exc)
        else:
            documents = chunk_transcript(transcript["snippets"], video_id)
            if not documents:
                error = "This video's transcript is empty, so there is nothing to index."

        if error:
            status.update(label="Could not load video", state="error", expanded=False)
        else:
            st.write("Building index")
            vector_store = build_vector_store(documents, get_embeddings())
            llm = get_llm(settings.groq_api_key, settings.groq_model)
            chain = build_chain(build_retriever(vector_store), llm)
            status.update(label="Video loaded", state="complete", expanded=False)

    # Shown outside the status box so the reason stays visible after the box collapses.
    if error:
        st.error(error, icon=":material/error:")
        return

    reset_session()
    st.session_state.video_id = video_id
    st.session_state.language = transcript["language"]
    st.session_state.duration = documents[-1].metadata["end"]
    st.session_state.vector_store = vector_store
    st.session_state.chain = chain


def render_message(role: str, content: str) -> None:
    with st.chat_message(role, avatar=AVATARS[role]):
        if role == "assistant":
            content = link_timestamps(content, st.session_state.video_id)
        st.markdown(content)


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.subheader(":material/smart_display: Load a video")
        # Enter to submit is off because its hint text overlaps long links in the input.
        with st.form("load_video", border=False, enter_to_submit=False):
            link = st.text_input(
                "YouTube link",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            submitted = st.form_submit_button(
                "Load video", type="primary", icon=":material/download:", width="stretch"
            )
        if submitted:
            load_video(link, settings)

        if st.session_state.video_id:
            with st.container(border=True):
                st.caption("Now chatting about")
                st.markdown(f":material/smart_display: `{st.session_state.video_id}`")
                st.markdown(f":material/closed_caption: {st.session_state.language}")
                left, right = st.columns(2)
                left.metric("Chunks", st.session_state.vector_store.index.ntotal)
                right.metric("Length", format_timestamp(st.session_state.duration))
                if st.button("Clear session", icon=":material/delete:", width="stretch"):
                    reset_session()
                    st.rerun()

        st.caption(
            "Chats are temporary. Refreshing or closing this tab clears the video and conversation."
        )


def render_empty_state() -> None:
    columns = st.columns(len(HOW_IT_WORKS))
    for column, (icon, heading, text) in zip(columns, HOW_IT_WORKS):
        with column.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{heading}**")
            st.caption(text)


st.set_page_config(page_title="AskTheVideo", page_icon=":material/smart_display:")
st.title("AskTheVideo")
st.caption("Ask questions about any YouTube video. Answers cite timestamps you can click.")

try:
    settings = load_settings()
except MissingApiKeyError:
    st.error(SETUP_STEPS, icon=":material/key:")
    st.stop()

if "video_id" not in st.session_state:
    reset_session()

render_sidebar(settings)

if st.session_state.video_id:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
else:
    render_empty_state()

for message in st.session_state.messages:
    render_message(message["role"], message["content"])

question = st.chat_input(
    "Ask a question about the video"
    if st.session_state.chain
    else "Load a video from the sidebar to start chatting",
    disabled=st.session_state.chain is None,
)
if question:
    render_message("user", question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the transcript"):
                answer = ask(st.session_state.chain, question)
        except AnswerError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            st.markdown(link_timestamps(answer, st.session_state.video_id))
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})
