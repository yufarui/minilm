from __future__ import annotations

from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

def _matched_files(local_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend([p for p in local_dir.rglob(pattern) if p.is_file()])
    # Deduplicate while preserving order.
    return list(dict.fromkeys(files))


def download_dataset_file(dataset_id: str, target_dir: Path, allow_file_patterns: list[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print("[1/3] Downloading files by ModelScope SDK...")
    local_dir = Path(
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_file_pattern=allow_file_patterns,
        )
    )
    print("[2/3] Verifying downloaded files...")
    files = _matched_files(local_dir, allow_file_patterns)
    if not files:
        visible_jsonl = sorted(str(p.relative_to(local_dir)) for p in local_dir.rglob("*.jsonl") if p.is_file())
        raise FileNotFoundError(
            "Download finished but no files matched the requested patterns: "
            f"{allow_file_patterns}. Please check dataset file names.\n"
            f"Visible *.jsonl under download dir: {visible_jsonl[:20]}"
        )
    print("[3/3] Done")
    print(f"  - Dataset: {dataset_id}")
    print(f"  - Download directory: {local_dir}")
    print("  - Matched files:")
    for file in files:
        print(f"    - {file}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target_dir = root / "data" / "pretrain"
    download_dataset_file(
        dataset_id="gongjy/minimind_dataset",
        target_dir=target_dir,
        # ModelScope's allow_file_pattern behaves closer to fnmatch than pathlib '**' glob.
        # Provide both direct-name and recursive patterns for better compatibility.
        allow_file_patterns=[
            "pretrain_t2t_mini.jsonl",
            "*pretrain_t2t_mini.jsonl",
            "**/pretrain_t2t_mini.jsonl",
        ],
    )


if __name__ == "__main__":
    main()
