from langchain_core.documents import Document


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
