import re

from langchain_core.documents import Document

# gpt-oss sometimes cites with its native full-width brackets (【04:32】) despite the prompt.
TIMESTAMP_RE = re.compile(r"[\[【]((?:(\d+):)?(\d{1,2}):(\d{2}))[\]】](?!\()")


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_context(documents: list[Document]) -> str:
    ordered = sorted(documents, key=lambda d: d.metadata["start"])
    return "\n\n".join(
        f"[{format_timestamp(d.metadata['start'])}] {d.page_content}" for d in ordered
    )


def link_timestamps(text: str, video_id: str) -> str:
    """Turn [mm:ss] and [h:mm:ss] citations into Markdown links that open the video at that moment."""

    def replace(match: re.Match) -> str:
        label, hours, minutes, secs = match.groups()
        seconds = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
        url = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
        return f"[[{label}]]({url})"

    return TIMESTAMP_RE.sub(replace, text)
