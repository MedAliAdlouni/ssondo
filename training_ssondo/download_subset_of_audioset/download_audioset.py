"""Download AudioSet audio clips from YouTube using yt-dlp."""

# Standard library imports
import os
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import pandas as pd
import yt_dlp

# Local imports
from training_ssondo import DATA
from .utils import (
    YTDLP_OPTS,
    get_output_filename,
    get_subdirectory,
    is_video_available,
)

SAMPLE_RATE = 16_000
AUDIO_CODEC = "pcm_s16le"
AUDIO_CHANNELS = 1


# ----------------------- Main entry point ---------------------------------------------------------
def download_subset_of_audioset(
    metadata_csv: str,
    subset_name: str,
    n_clips: int = 5,
    max_workers: int = 5,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> dict:
    """Download a subset of AudioSet clips from YouTube (keeps trying until it gets enough)."""
    if output_dir is None:
        data_root = DATA
        output_dir = os.path.join(data_root, "AudioSet")

    df = pd.read_csv(
        metadata_csv,
        skiprows=2,
        names=["file_id", "start_seconds", "end_seconds", "positive_labels"],
        skipinitialspace=True,
        quotechar='"',
        low_memory=False,
    )
    print(f"Loaded {len(df)} clips from metadata.")

    return download_audio_segments_parallel(
        df, n_clips, output_dir, subset_name, random_state, max_workers
    )


# ----------------------- Download audio segments in parallel ---------------------------------------
def download_audio_segments_parallel(
    full_df: pd.DataFrame,
    n_clips_wanted: int,
    output_dir: str,
    subset_name: str,
    random_state: int = 42,
    max_workers: int = 5,
) -> dict:
    """Download clips until we get `n_clips_wanted` successes."""
    subset_dir = Path(os.path.join(output_dir, subset_name))
    subset_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "skipped_unavailable": 0,
        "failed_video_ids": [],
        "successful_video_ids": [],
    }
    lock = threading.Lock()

    df = full_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(
        f"\nAttempting {n_clips_wanted} clips from {len(df)} rows with {max_workers} workers.\n"
    )

    def worker(i: int) -> None:
        row = df.iloc[i]
        vid, start, end = (
            str(row["file_id"]),
            float(row["start_seconds"]),
            float(row["end_seconds"]),
        )
        out = subset_dir / get_subdirectory(vid) / get_output_filename(vid, start, end)

        # Check if already exists
        if out.exists():
            with lock:
                stats["successful"] += 1
                stats["successful_video_ids"].append(vid)
                s = stats["successful"]
            print(f"[{s}/{n_clips_wanted}] exists {vid}")
            return

        # Check if video is available
        if not is_video_available(vid):
            with lock:
                stats["skipped_unavailable"] += 1
            print(f"[{i + 1}] skip unavailable {vid}")
            return

        # Attempt download
        with lock:
            stats["total_attempted"] += 1
            s = stats["successful"]
        print(f"[{s}/{n_clips_wanted}] downloading {vid} @ {start}s")

        ok = download_audio_segment(vid, str(out), start, end - start)
        with lock:
            if ok:
                stats["successful"] += 1
                stats["successful_video_ids"].append(vid)
            else:
                stats["failed"] += 1
                stats["failed_video_ids"].append(vid)
            s2 = stats["successful"]
        print(("  ✓ " if ok else "  ✗ ") + f"{vid} ({s2}/{n_clips_wanted})")

    # Process with thread pool
    futures = {}
    next_i = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit initial batch
            while next_i < len(df) and len(futures) < max_workers:
                futures[ex.submit(worker, next_i)] = next_i
                next_i += 1

            # Process until we have enough or run out of rows
            while futures:
                done_fut = next(as_completed(list(futures.keys())), None)
                if done_fut is None:
                    break

                done_fut.result()  # propagate exceptions
                del futures[done_fut]

                # Check if we have enough BEFORE submitting more tasks
                with lock:
                    if stats["successful"] >= n_clips_wanted:
                        for f in futures:
                            f.cancel()
                        break

                # Only submit next task if we haven't reached the target yet
                with lock:
                    if stats["successful"] < n_clips_wanted and next_i < len(df):
                        futures[ex.submit(worker, next_i)] = next_i
                        next_i += 1

            # Cancel remaining
            for f in futures:
                f.cancel()
    except KeyboardInterrupt:
        print("\nInterrupted. Canceling pending tasks...")
        for f in futures:
            f.cancel()

    # Print statistics
    scanned = next_i

    print(f"\n{'=' * 60}\nDOWNLOAD COMPLETE\n{'=' * 60}")
    print(f"Successful downloads: {stats['successful']}/{n_clips_wanted}")
    print(f"Scanned rows: {scanned}/{len(df)}")
    print(f"  - Skipped (unavailable): {stats['skipped_unavailable']}")
    print(f"  - Download attempted: {stats['total_attempted']}")
    print(f"  - Failed: {stats['failed']}")
    if stats["successful"] < n_clips_wanted:
        print(
            f"\nWARNING: only got {stats['successful']} of {n_clips_wanted}. Not enough available clips in dataset OR The download was interrupted."
        )
    print(f"{'=' * 60}")

    return stats


# ----------------- Download a single audio segment using yt-dlp ------------------------------------------
def download_audio_segment(
    video_id: str,
    output_path: str,
    start_time: float,
    duration: float = 10.0,
) -> bool:
    """Download and extract a mono 16kHz WAV segment from YouTube."""
    outp = Path(output_path)
    if outp.exists():
        return True

    temp_dir = outp.parent / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_pattern = temp_dir / f"{video_id}_temp"

    try:
        # Download video
        with yt_dlp.YoutubeDL(
            {
                "format": "bestaudio/best",
                "outtmpl": str(temp_pattern) + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                **YTDLP_OPTS,
            }
        ) as ydl:
            ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=True
            )

        # Find downloaded file and extract segment
        matches = sorted(temp_dir.glob(f"{video_id}_temp.*"))
        if not matches:
            return False

        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                str(matches[0]),
                "-acodec",
                AUDIO_CODEC,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                str(AUDIO_CHANNELS),
                str(outp),
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and outp.exists()
    except Exception:
        traceback.print_exc()
        return False
    finally:
        # Cleanup: remove temp files
        for p in temp_dir.glob(f"{video_id}_temp.*"):
            p.unlink(missing_ok=True)  # Python 3.8+ - handles exceptions automatically


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Download AudioSet audio clips from YouTube"
    )
    p.add_argument("--metadata-csv", required=True, help="Path to metadata CSV")
    p.add_argument(
        "--n-clips",
        type=int,
        default=5,
        help="Number of clips to successfully download",
    )
    p.add_argument(
        "--subset-name",
        required=True,
        help="Output subdirectory name (e.g. 'eval', 'balanced_train', 'unbalanced_train')",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for downloaded AudioSet audio files (default: $DATA/AudioSet or training_ssondo/data/AudioSet)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument(
        "--max-workers", type=int, default=5, help="Parallel download workers"
    )
    args = p.parse_args()

    stats = download_subset_of_audioset(
        metadata_csv=args.metadata_csv,
        n_clips=args.n_clips,
        random_state=args.random_state,
        subset_name=args.subset_name,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
    )
