import pytest
import requests
from youtube_transcript_api import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
    IpBlocked,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from askthevideo import youtube
from askthevideo.youtube import (
    InvalidVideoLinkError,
    TranscriptError,
    fetch_transcript,
    parse_video_id,
)

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890&index=3",
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc123&t=10",
        "youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ?feature=share",
        "https://m.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ?start=30",
        "dQw4w9WgXcQ",
        "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",
    ],
)
def test_parse_valid_links(link):
    assert parse_video_id(link) == VIDEO_ID


def test_parse_id_with_underscore_and_dash():
    assert parse_video_id("https://youtu.be/a_b-C_d-E_f") == "a_b-C_d-E_f"


@pytest.mark.parametrize(
    "link",
    [
        "",
        "   ",
        "hello world",
        "dQw4w9WgXc",
        "dQw4w9WgXcQQ",
        "dQw4w9WgX!Q",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?list=PL1234567890",
        "https://www.youtube.com/channel/UC1234567890",
        "https://www.youtube.com/",
        "https://youtu.be/",
        "https://youtu.be/dQw4w9WgXcQ/extra",
        "https://www.youtube.com/shorts/",
        "https://vimeo.com/watch?v=dQw4w9WgXcQ",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_parse_invalid_links(link):
    with pytest.raises(InvalidVideoLinkError):
        parse_video_id(link)


class FakeTranscript:
    def __init__(self, language_code, language):
        self.language_code = language_code
        self.language = language

    def fetch(self):
        return FetchedTranscript(
            snippets=[FetchedTranscriptSnippet(text=self.language, start=1.5, duration=2.0)],
            video_id=VIDEO_ID,
            language=self.language,
            language_code=self.language_code,
            is_generated=False,
        )


class FakeApi:
    def __init__(self, result):
        self.result = result

    def list(self, video_id):
        if isinstance(self.result, Exception):
            raise self.result
        return iter(self.result)


def use_fake_api(monkeypatch, result):
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda: FakeApi(result))


def test_fetch_returns_language_and_plain_dicts(monkeypatch):
    use_fake_api(monkeypatch, [FakeTranscript("en", "English")])
    assert fetch_transcript(VIDEO_ID) == {
        "language": "English",
        "snippets": [{"text": "English", "start": 1.5, "duration": 2.0}],
    }


def test_fetch_prefers_english_over_earlier_languages(monkeypatch):
    transcripts = [
        FakeTranscript("de", "German"),
        FakeTranscript("en-GB", "English (United Kingdom)"),
    ]
    use_fake_api(monkeypatch, transcripts)
    assert fetch_transcript(VIDEO_ID)["language"] == "English (United Kingdom)"


def test_fetch_falls_back_to_first_available_language(monkeypatch):
    transcripts = [
        FakeTranscript("hi", "Hindi (auto-generated)"),
        FakeTranscript("es", "Spanish"),
    ]
    use_fake_api(monkeypatch, transcripts)
    assert fetch_transcript(VIDEO_ID)["language"] == "Hindi (auto-generated)"


def test_fetch_does_not_treat_other_en_prefixes_as_english(monkeypatch):
    transcripts = [FakeTranscript("hi", "Hindi"), FakeTranscript("enm", "Middle English")]
    use_fake_api(monkeypatch, transcripts)
    assert fetch_transcript(VIDEO_ID)["language"] == "Hindi"


def test_fetch_with_no_transcripts(monkeypatch):
    use_fake_api(monkeypatch, [])
    with pytest.raises(TranscriptError, match="no captions in any language"):
        fetch_transcript(VIDEO_ID)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TranscriptsDisabled(VIDEO_ID), "no captions in any language"),
        (VideoUnavailable(VIDEO_ID), "unavailable"),
        (RequestBlocked(VIDEO_ID), "blocked transcript requests"),
        (IpBlocked(VIDEO_ID), "blocked transcript requests"),
        (requests.ConnectionError(), "Could not reach YouTube"),
    ],
)
def test_fetch_maps_errors(monkeypatch, error, expected):
    use_fake_api(monkeypatch, error)
    with pytest.raises(TranscriptError, match=expected):
        fetch_transcript(VIDEO_ID)
