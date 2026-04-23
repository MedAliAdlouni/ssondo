"""Generate metadata.csv from AudioSet segment CSVs and ontology.

Combines eval_segments.csv, balanced_train_segments.csv, and
unbalanced_train_segments.csv with ontology.json to produce the unified
metadata.csv expected by the training pipeline.

Usage:
    uv run python scripts/generate_metadata.py [--data-dir data/AudioSet]
"""

import argparse
import csv
import json
import os
import sys


def get_subdirectory(video_id: str) -> str:
    return "-" if video_id[0] in ("-", "_", ".") else video_id[0]


def load_ontology(ontology_path: str) -> dict:
    with open(ontology_path) as f:
        ontology = json.load(f)
    return {entry["id"]: entry["name"] for entry in ontology}


def load_class_labels(class_labels_path: str) -> dict:
    mid_to_idx = {}
    with open(class_labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid_to_idx[row["mid"]] = int(row["index"])
    return mid_to_idx


def parse_segment_csv(path: str):
    """Parse an AudioSet segment CSV (skips comment lines starting with #)."""
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(", ")
            if len(parts) < 4:
                continue
            video_id = parts[0]
            start = float(parts[1])
            end = float(parts[2])
            labels_str = parts[3].strip().strip('"')
            label_mids = [mid.strip() for mid in labels_str.split(",")]
            rows.append((video_id, start, end, label_mids))
    return rows


def generate_metadata(data_dir: str, output_path: str):
    ontology_path = os.path.join(data_dir, "ontology.json")
    class_labels_path = os.path.join(data_dir, "class_labels_indices.csv")

    for required in [ontology_path, class_labels_path]:
        if not os.path.exists(required):
            print(f"Error: Required file not found: {required}")
            sys.exit(1)

    mid_to_name = load_ontology(ontology_path)
    mid_to_idx = load_class_labels(class_labels_path)

    segment_files = {
        "eval": "eval_segments.csv",
        "balanced_train": "balanced_train_segments.csv",
        "unbalanced_train": "unbalanced_train_segments.csv",
    }

    columns = [
        "",
        "file_id",
        "file_path",
        "set",
        "start_seconds",
        "end_seconds",
        "label",
        "label_idx",
        "strong_annot",
        "n_channels",
        "sampling_rate",
        "duration",
        "cutoff_freq",
        "video_file_path",
    ]

    row_idx = 0
    total_rows = 0

    with open(output_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(columns)

        for subset, filename in segment_files.items():
            segment_path = os.path.join(data_dir, filename)
            if not os.path.exists(segment_path):
                print(f"  Warning: {filename} not found, skipping {subset}")
                continue

            segments = parse_segment_csv(segment_path)
            print(f"  Processing {subset}: {len(segments)} clips")

            for video_id, start, end, label_mids in segments:
                subdir = get_subdirectory(video_id)
                file_path = f"{subset}/{subdir}/{video_id}_{start:.3f}_{end:.3f}.wav"

                for mid in label_mids:
                    label_name = mid_to_name.get(mid, mid)
                    label_idx = mid_to_idx.get(mid, -1)

                    writer.writerow([
                        row_idx,
                        video_id,
                        file_path,
                        subset,
                        start,
                        end,
                        label_name,
                        label_idx,
                        "",
                        1,
                        16000,
                        10.0,
                        8000,
                        "",
                    ])
                    row_idx += 1

            total_rows += len(segments)

    print(f"  Generated {row_idx} rows ({total_rows} clips) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AudioSet metadata.csv")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "AudioSet"),
        help="Path to AudioSet data directory",
    )
    args = parser.parse_args()

    output_path = os.path.join(args.data_dir, "metadata.csv")
    print(f"Generating metadata.csv in {args.data_dir}...")
    generate_metadata(args.data_dir, output_path)
