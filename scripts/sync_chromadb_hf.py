#!/usr/bin/env python3
"""
Sync local ChromaDB with a Hugging Face dataset repo.

Usage:
  python3 scripts/sync_chromadb_hf.py status
  python3 scripts/sync_chromadb_hf.py push --repo-id Agnes999/legalbot9
  python3 scripts/sync_chromadb_hf.py pull --repo-id Agnes999/legalbot9
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, upload_folder
except Exception as e:
    print(f"❌ huggingface_hub is required: {e}")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def resolve_persist_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_dir = (os.getenv("CHROMA_PERSIST_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "chroma_db").resolve()


def resolve_repo_id(raw: str | None) -> str:
    repo_id = (raw or os.getenv("CHROMA_HF_DATASET_REPO") or "").strip()
    if not repo_id:
        raise ValueError("Missing repo id. Provide --repo-id or CHROMA_HF_DATASET_REPO.")
    return repo_id


def resolve_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def local_db_stats(persist_dir: Path) -> tuple[bool, int, int]:
    if not persist_dir.exists():
        return False, 0, 0
    total_size = 0
    total_files = 0
    for root, _dirs, files in os.walk(persist_dir):
        for name in files:
            p = Path(root) / name
            try:
                total_size += p.stat().st_size
                total_files += 1
            except OSError:
                pass
    sqlite_ok = (persist_dir / "chroma.sqlite3").exists()
    return sqlite_ok, total_files, total_size


def cmd_status(args: argparse.Namespace) -> int:
    persist_dir = resolve_persist_dir(args.persist_dir)
    sqlite_ok, files, total_size = local_db_stats(persist_dir)
    print(f"📁 Persist dir: {persist_dir}")
    print(f"🧩 SQLite present: {sqlite_ok}")
    print(f"📄 Files: {files}")
    print(f"💾 Size: {format_size(total_size)}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    persist_dir = resolve_persist_dir(args.persist_dir)
    repo_id = resolve_repo_id(args.repo_id)
    token = resolve_token()

    if not persist_dir.exists():
        print(f"❌ Local persist dir not found: {persist_dir}")
        return 1
    if not (persist_dir / "chroma.sqlite3").exists():
        print(f"❌ Missing chroma.sqlite3 in: {persist_dir}")
        return 1

    print(f"⬆️ Uploading ChromaDB to HF dataset: {repo_id}")
    print(f"   Source: {persist_dir}")
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(persist_dir),
        path_in_repo="",
        token=token,
        commit_message=args.commit_message,
        ignore_patterns=[".DS_Store", "__pycache__/*", "*.tmp", "*.swp"],
    )
    print("✅ Upload complete")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    persist_dir = resolve_persist_dir(args.persist_dir)
    repo_id = resolve_repo_id(args.repo_id)
    token = resolve_token()
    revision = args.revision or (os.getenv("CHROMA_HF_REVISION") or "").strip() or None

    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading ChromaDB from HF dataset: {repo_id}")
    print(f"   Destination: {persist_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(persist_dir),
        token=token,
        revision=revision,
    )
    print("✅ Download complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync ChromaDB with HF dataset.")
    parser.add_argument("--persist-dir", help="Local ChromaDB directory (default: CHROMA_PERSIST_DIR or ./chroma_db)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show local ChromaDB status")
    p_status.set_defaults(func=cmd_status)

    p_push = sub.add_parser("push", help="Upload local ChromaDB to HF dataset")
    p_push.add_argument("--repo-id", help="HF dataset repo id, e.g. Agnes999/legalbot9")
    p_push.add_argument("--commit-message", default="Upload ChromaDB snapshot")
    p_push.set_defaults(func=cmd_push)

    p_pull = sub.add_parser("pull", help="Download ChromaDB from HF dataset")
    p_pull.add_argument("--repo-id", help="HF dataset repo id, e.g. Agnes999/legalbot9")
    p_pull.add_argument("--revision", help="Optional branch/tag/commit")
    p_pull.set_defaults(func=cmd_pull)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
