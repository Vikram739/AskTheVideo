from langchain_core.documents import Document

WINDOW_SECONDS = 60.0
OVERLAP_SECONDS = 10.0


def chunk_transcript(
    snippets: list[dict],
    video_id: str,
    window: float = WINDOW_SECONDS,
    overlap: float = OVERLAP_SECONDS,
) -> list[Document]:
    """Group snippets into time windows; a snippet belongs to every window its start falls in."""
    snippets = sorted(
        (s for s in snippets if s["text"].strip()), key=lambda s: s["start"]
    )
    if not snippets:
        return []

    step = window - overlap
    documents = []
    window_start = snippets[0]["start"]
    while True:
        window_end = window_start + window
        members = [s for s in snippets if window_start <= s["start"] < window_end]
        documents.append(
            Document(
                page_content=" ".join(" ".join(s["text"].split()) for s in members),
                metadata={
                    "video_id": video_id,
                    "start": members[0]["start"],
                    "end": max(s["start"] + s["duration"] for s in members),
                },
            )
        )

        remaining = [s for s in snippets if s["start"] >= window_end]
        if not remaining:
            return documents
        window_start += step
        # Skip silent gaps so a window never repeats only snippets the previous one already covered.
        if remaining[0]["start"] >= window_start + window:
            window_start = remaining[0]["start"]
