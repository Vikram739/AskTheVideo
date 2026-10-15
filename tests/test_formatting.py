import pytest
from langchain_core.documents import Document

from askthevideo.formatting import format_context, format_timestamp, link_timestamps


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (5.9, "00:05"),
        (65, "01:05"),
        (272, "04:32"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3725.4, "1:02:05"),
        (36000, "10:00:00"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_context_orders_by_start_and_separates_with_blank_line():
    docs = [
        Document(page_content="second", metadata={"start": 3725.0}),
        Document(page_content="first", metadata={"start": 272.0}),
    ]
    assert format_context(docs) == "[04:32] first\n\n[1:02:05] second"


def test_format_context_empty():
    assert format_context([]) == ""


def test_link_minutes_seconds():
    assert link_timestamps("See [04:32].", "VID") == (
        "See [[04:32]](https://www.youtube.com/watch?v=VID&t=272s)."
    )


def test_link_hours_minutes_seconds():
    assert link_timestamps("[1:02:05]", "VID") == (
        "[[1:02:05]](https://www.youtube.com/watch?v=VID&t=3725s)"
    )


def test_link_multiple_timestamps():
    text = "- a [00:10]\n- b [01:00]"
    assert link_timestamps(text, "VID") == (
        "- a [[00:10]](https://www.youtube.com/watch?v=VID&t=10s)\n"
        "- b [[01:00]](https://www.youtube.com/watch?v=VID&t=60s)"
    )


def test_link_full_width_brackets_render_as_square_brackets():
    assert link_timestamps("pixels【02:37】.", "VID") == (
        "pixels[[02:37]](https://www.youtube.com/watch?v=VID&t=157s)."
    )


def test_link_zero_padded_hours():
    assert link_timestamps("[01:11:41]", "VID") == (
        "[[01:11:41]](https://www.youtube.com/watch?v=VID&t=4301s)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "No timestamps here.",
        "04:32 without brackets",
        "[note] and [4]",
        "[04:32](https://example.com) already a link",
    ],
)
def test_link_leaves_other_text_unchanged(text):
    assert link_timestamps(text, "VID") == text
