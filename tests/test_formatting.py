import pytest
from langchain_core.documents import Document

from askthevideo.formatting import format_context, format_timestamp


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
