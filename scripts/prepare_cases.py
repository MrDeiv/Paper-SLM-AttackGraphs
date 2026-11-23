#!/usr/bin/env python3
"""
Prepare NCSC case-study folders from the repository's ZIP archives.

The current repository stores documents as documents/<case>.zip while the legacy
config expects extracted directories. This script extracts each archive into a
stable data/cases/<case_id>/documents directory and writes a manifest consumed by
experiment runners.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


CASE_TO_STIX = {
    "authentic": "sample_stix/ncsc-mar-authentic-antics-stix.json",
    "chipmunk": "sample_stix/NCSC-MAR-Cheeky-Chipmunk-stix.json",
    "coldsteel": "sample_stix/NCSC-MAR-Cold-Steel-stix.json",
    "damascened": "sample_stix/ncsc-mar-damascened-peacock-stix.json",
    "goofy": "sample_stix/NCSC-MAR-goofy-guineapig-stix.json",
    "jaguar": "sample_stix/NCSC-MAR-Jaguar-Tooth-stix.json",
    "pygmy": "sample_stix/ncsc-mar-pygmy-goat-stix.json",
    "small_sieve": "sample_stix/NCSC-MAR-Small-Sieve-stix.json",
    "smooth": "sample_stix/NCSC-MAR-Smooth-Operator-stix.json",
    "umbrella": "sample_stix/ncsc-mar-umbrella-stand-stix.json",
}

CASE_TO_GROUND_TRUTH = {
    "authentic": "ground_truths/authentic.json",
    "chipmunk": "ground_truths/chipmunk.json",
    "coldsteel": "ground_truths/coldsteel.json",
    "damascened": "ground_truths/damascened.json",
    "goofy": "ground_truths/goofy.json",
    "jaguar": "ground_truths/jaguar.json",
    "pygmy": "ground_truths/pygmy.json",
    "small_sieve": "ground_truths/sieve.json",
    "smooth": "ground_truths/smooth.json",
    "umbrella": "ground_truths/umbrella.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract case ZIP files and write a case manifest.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--documents-dir", default="documents", help="Directory containing case ZIP files.")
    parser.add_argument("--out-dir", default="data/cases", help="Output directory for extracted case folders.")
    parser.add_argument("--manifest", default="data/case_manifest.json", help="Output manifest path.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract archives even if output exists.")
    return parser.parse_args()


def safe_extract(zip_file: zipfile.ZipFile, destination: Path) -> None:
    """
    Extract a ZIP archive while preventing path traversal.
    """
    destination = destination.resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if not str(target).startswith(str(destination)):
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    zip_file.extractall(destination)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    documents_dir = repo_root / args.documents_dir
    out_dir = repo_root / args.out_dir
    manifest_path = repo_root / args.manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    for archive_path in sorted(documents_dir.glob("*.zip")):
        case_id = archive_path.stem
        if case_id == "sparrow":
            # The uploaded repository has sparrow.zip but no matching STIX/ground-truth entry.
            continue

        case_dir = out_dir / case_id
        docs_dir = case_dir / "documents"
        if docs_dir.exists() and args.overwrite:
            for child in docs_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)

        docs_dir.mkdir(parents=True, exist_ok=True)
        if not any(docs_dir.iterdir()) or args.overwrite:
            with zipfile.ZipFile(archive_path, "r") as zf:
                safe_extract(zf, docs_dir)

        stix_file = CASE_TO_STIX.get(case_id)
        ground_truth = CASE_TO_GROUND_TRUTH.get(case_id)

        manifest.append(
            {
                "case_id": case_id,
                "archive": str(archive_path.relative_to(repo_root)),
                "documents_dir": str(docs_dir.relative_to(repo_root)),
                "stix_file": stix_file,
                "ground_truth": ground_truth,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Prepared {len(manifest)} cases")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()