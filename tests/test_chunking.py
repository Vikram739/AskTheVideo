from askthevideo.chunking import chunk_transcript


def snippets_every(seconds, count, duration=5.0):
    return [
        {"text": f"s{i}", "start": float(i * seconds), "duration": duration}
        for i in range(count)
    ]


def test_empty_transcript():
    assert chunk_transcript([], "vid") == []


def test_blank_snippets_are_ignored():
    assert chunk_transcript([{"text": " \n", "start": 0.0, "duration": 1.0}], "vid") == []


def test_single_snippet():
    docs = chunk_transcript([{"text": "only one", "start": 3.0, "duration": 4.0}], "vid")
    assert len(docs) == 1
    assert docs[0].page_content == "only one"
    assert docs[0].metadata == {"video_id": "vid", "start": 3.0, "end": 7.0}


def test_window_boundaries_and_overlap():
    # Snippets at 0, 5, ..., 145 seconds. Windows: [0,60), [50,110), [100,160).
    docs = chunk_transcript(snippets_every(5, 30), "vid")
    starts = [[int(w[1:]) * 5 for w in d.page_content.split()] for d in docs]
    assert starts == [
        list(range(0, 60, 5)),
        list(range(50, 110, 5)),
        list(range(100, 150, 5)),
    ]
    # Snippets in the last 10 seconds of a window repeat at the start of the next one.
    assert starts[0][-2:] == starts[1][:2] == [50, 55]


def test_metadata_values():
    docs = chunk_transcript(snippets_every(5, 30, duration=4.0), "abc123XYZ_-")
    assert [d.metadata for d in docs] == [
        {"video_id": "abc123XYZ_-", "start": 0.0, "end": 59.0},
        {"video_id": "abc123XYZ_-", "start": 50.0, "end": 109.0},
        {"video_id": "abc123XYZ_-", "start": 100.0, "end": 149.0},
    ]


def test_end_uses_longest_snippet_in_window():
    snippets = [
        {"text": "long", "start": 0.0, "duration": 30.0},
        {"text": "short", "start": 10.0, "duration": 2.0},
    ]
    assert chunk_transcript(snippets, "vid")[0].metadata["end"] == 30.0


def test_unsorted_input_is_ordered_and_whitespace_collapsed():
    snippets = [
        {"text": "world", "start": 2.0, "duration": 1.0},
        {"text": "hello\nthere ", "start": 0.0, "duration": 1.0},
    ]
    assert chunk_transcript(snippets, "vid")[0].page_content == "hello there world"


def test_gap_skips_to_next_snippet_without_duplicate_chunk():
    snippets = [
        {"text": "a", "start": 0.0, "duration": 2.0},
        {"text": "b", "start": 55.0, "duration": 2.0},
        {"text": "c", "start": 300.0, "duration": 2.0},
    ]
    docs = chunk_transcript(snippets, "vid")
    assert [d.page_content for d in docs] == ["a b", "c"]
    assert docs[1].metadata["start"] == 300.0


def test_every_snippet_is_covered():
    snippets = snippets_every(7, 100)
    docs = chunk_transcript(snippets, "vid")
    covered = {w for d in docs for w in d.page_content.split()}
    assert covered == {s["text"] for s in snippets}
