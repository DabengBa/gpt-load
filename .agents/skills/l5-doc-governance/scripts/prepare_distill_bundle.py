#!/usr/bin/env python3
"""Collect worktree .tmp/ ephemeral docs, branch diff, and durable-owner index for pre-PR distillation."""
import argparse
import json
import subprocess
from pathlib import Path

DOC_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".log"}
MAX_FILE_CHARS = 12000


def run_git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""


def existing_owner_files(worktree: Path) -> list[str]:
    """Repos with an established durable-owner tree need the same dedup index."""
    found: list[str] = []
    for rel in (".docs/adr", ".docs/tech", ".docs/db"):
        d = worktree / rel
        if d.exists():
            found.extend(str(p.relative_to(worktree)) for p in sorted(d.rglob("*.md")))
    return found[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle ephemeral .tmp/ docs and diff for Distiller Agent.")
    parser.add_argument("--worktree", default=".", help="Path to active worktree root.")
    parser.add_argument("--base-ref", default="origin/main", help="Base git ref for PR branch diff.")
    parser.add_argument("--max-diff-chars", type=int, default=24000)
    parser.add_argument("--out", default="", help="Optional output file path (defaults to stdout).")
    args = parser.parse_args()

    wt = Path(args.worktree).resolve()
    tmp_dir = wt / ".tmp"
    ephemeral_docs = []
    if tmp_dir.exists():
        for p in sorted(tmp_dir.rglob("*")):
            if p.is_file() and p.name != "distill-bundle.json" and p.suffix.lower() in DOC_SUFFIXES:
                content = p.read_text(encoding="utf-8", errors="replace")
                ephemeral_docs.append({
                    "path": str(p.relative_to(wt)),
                    "content": content[:MAX_FILE_CHARS],
                    "truncated": len(content) > MAX_FILE_CHARS,
                })

    diff_range = f"{args.base_ref}...HEAD"
    diff_stat = run_git(wt, ["diff", "--stat", diff_range])
    if not diff_stat:
        diff_range = args.base_ref
        diff_stat = run_git(wt, ["diff", "--stat", diff_range])
    diff_full = run_git(wt, ["diff", "--unified=3", diff_range])
    working_diff = run_git(wt, ["diff", "--unified=3"])
    if working_diff:
        diff_full = (diff_full + "\n# --- UNCOMMITTED WORKTREE DIFF ---\n" + working_diff).strip()
    if len(diff_full) > args.max_diff_chars:
        diff_full = diff_full[: args.max_diff_chars] + "\n...[DIFF TRUNCATED]..."

    existing_knowledge = {}
    kdir = wt / ".knowledge"
    if kdir.exists():
        for cat in ("decisions", "invariants", "quirks"):
            cat_dir = kdir / cat
            existing_knowledge[cat] = [f.name for f in sorted(cat_dir.glob("*.md"))] if cat_dir.exists() else []

    bundle = {
        "worktree": str(wt),
        "base_ref": args.base_ref,
        "ephemeral_tmp_files": ephemeral_docs,
        "git_diff_stat": diff_stat,
        "git_diff": diff_full,
        "existing_knowledge_index": existing_knowledge,
        "existing_owner_files": existing_owner_files(wt),
    }
    payload = json.dumps(bundle, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(json.dumps({
            "status": "ok",
            "bundle_path": str(out_path),
            "ephemeral_files": len(ephemeral_docs),
            "diff_chars": len(diff_full),
            "existing_owner_files": len(bundle["existing_owner_files"]),
        }, ensure_ascii=False))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
