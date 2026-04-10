from __future__ import annotations

from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

def download_dataset_file(dataset_id: str, target_file_path: Path, allow_file_pattern: str) :
    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("[1/2] Downloading file by ModelScope SDK...")
    local_dir = Path(
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            local_dir=str(target_file_path),
            allow_file_pattern=allow_file_pattern,
        )
    )
    print("[2/2] Done")
    print(f"  - Dataset: {dataset_id}")
    print(f"  - Download directory: {local_dir}")
    print(f"  - Target file name: {target_file_path.name}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target_file_path = root / "data" / "pretrain"
    download_dataset_file(
        dataset_id="gongjy/minimind_dataset",
        target_file_path=target_file_path,
        allow_file_pattern=f"**/pretrain_t2t.jsonl",
    )


if __name__ == "__main__":
    main()
