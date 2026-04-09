"""Utility functions for downloading AudioSet audio clips from YouTube."""

import yt_dlp


YTDLP_OPTS = {
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-us,en;q=0.5",
        "Sec-Fetch-Mode": "navigate",
    },
    "extractor_args": {"youtube": {"player_client": ["android", "tv", "web"], "skip": ["dash", "hls"]}},
    "retries": 10,
    "fragment_retries": 10,
    "socket_timeout": 30,
}

def get_subdirectory(video_id: str) -> str:
    """Get subdirectory name based on video_id (first character as subdirectory)"""
    return "-" if video_id[0] in ['-', '_', '.'] else video_id[0]

def get_output_filename(video_id: str, start_time: float, end_time: float) -> str:
    """Generate output filename in the correct format."""
    return f"{video_id}_{start_time:.3f}_{end_time:.3f}.wav"

def is_video_available(video_id: str) -> bool:
    """Check if YouTube video is available without downloading."""
    try:
        opts = {"quiet": True, "no_warnings": True, "extract_flat": True, **YTDLP_OPTS}
        with yt_dlp.YoutubeDL(opts) as ydl:
            return bool(ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False))
    except Exception:
        return False
