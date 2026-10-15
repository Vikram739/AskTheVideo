import re
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApi,
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

INVALID_LINK_MESSAGE = (
    "That doesn't look like a YouTube video link. Paste a link such as "
    "https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID."
)

NO_CAPTIONS_MESSAGE = (
    "This video has no captions in any language, so it can't be indexed. "
    "Try a video that shows the CC button on YouTube."
)


class InvalidVideoLinkError(ValueError):
    pass


class TranscriptError(Exception):
    pass


def parse_video_id(link: str) -> str:
    link = link.strip()
    if VIDEO_ID_RE.match(link):
        return link

    if "://" not in link:
        link = "https://" + link
    parsed = urlparse(link)
    if parsed.scheme not in ("http", "https"):
        raise InvalidVideoLinkError(INVALID_LINK_MESSAGE)

    host = (parsed.hostname or "").lower()
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)

    segments = [s for s in parsed.path.split("/") if s]
    candidate = None
    if host == "youtu.be" and len(segments) == 1:
        candidate = segments[0]
    elif host == "youtube.com":
        if segments == ["watch"]:
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(segments) == 2 and segments[0] in ("shorts", "embed"):
            candidate = segments[1]

    if candidate is None or not VIDEO_ID_RE.match(candidate):
        raise InvalidVideoLinkError(INVALID_LINK_MESSAGE)
    return candidate


def _is_english(language_code: str) -> bool:
    return language_code == "en" or language_code.startswith("en-")


def fetch_transcript(video_id: str) -> dict:
    """Return the transcript language name and its snippets (dicts with text, start, duration).

    English is preferred. Other languages are accepted because the multilingual
    embedding model matches English questions to non-English transcripts.
    """
    try:
        # The list yields manually created transcripts before auto-generated ones.
        transcripts = list(YouTubeTranscriptApi().list(video_id))
        if not transcripts:
            raise TranscriptError(NO_CAPTIONS_MESSAGE)
        transcript = next(
            (t for t in transcripts if _is_english(t.language_code)), transcripts[0]
        )
        fetched = transcript.fetch()
    except RequestBlocked as exc:
        # IpBlocked subclasses RequestBlocked, so both land here.
        raise TranscriptError(
            "YouTube blocked transcript requests from this network. This usually "
            "happens on cloud or VPN IP addresses. Run the app from a home network "
            "and try again."
        ) from exc
    except TranscriptsDisabled as exc:
        raise TranscriptError(NO_CAPTIONS_MESSAGE) from exc
    except (VideoUnavailable, InvalidVideoId) as exc:
        raise TranscriptError(
            "This video is unavailable. Check the link, or the video may be private or removed."
        ) from exc
    except (AgeRestricted, VideoUnplayable) as exc:
        raise TranscriptError(
            "This video can't be played without signing in (for example, it is "
            "age restricted or members only), so its transcript can't be fetched."
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(
            "Could not fetch the transcript for this video. Try again later."
        ) from exc
    except requests.RequestException as exc:
        raise TranscriptError(
            "Could not reach YouTube. Check your internet connection and try again."
        ) from exc
    return {
        "language": transcript.language,
        "snippets": fetched.to_raw_data(),
    }
