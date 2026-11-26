#!/usr/bin/env python3
"""Initialize L5 pure-agent documentation structure and ephemeral .tmp/ rules."""
import argparse
import json
import shutil
from pathlib import Path


def find_root_agents(repo_root: Path) -> Path | None:
    """Root boot contract may be AGENTS.md, AGENTS.MD, or any case variant."""
    return next(
        (p for p in sorted(repo_root.glob("*")) if p.is_file() and p.name.lower() == "agents.md"),
        None,
    )


def ensure_gitignore(repo_root: Path, patterns: list[str]) -> list[str]:
    gitignore = repo_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    lines = [line.strip() for line in existing.splitlines()]
    added = [pat for pat in patterns if pat not in lines]
    if added:
        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        block = (
            "# L5 Ephemeral worktree process workspace (never commit or merge)\n"
            + "\n".join(added)
            + "\n"
        )
        gitignore.write_text(existing + prefix + block, encoding="utf-8")
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize L5 documentation architecture.")
    parser.add_argument("--repo", default=".", help="Path to target production repository root.")
    parser.add_argument("--with-templates", action="store_true", help="Create starter AGENTS.md if missing.")
    parser.add_argument("--init-holdout", default="", help="Optional path to initialize Space C Clean-Room Holdout repo.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing AGENTS.md when --with-templates is set.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    created: list[str] = []

    dirs = [
        repo / ".knowledge" / "decisions",
        repo / ".knowledge" / "invariants",
        repo / ".knowledge" / "quirks",
        repo / ".agents" / "skills",
    ]
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d.relative_to(repo)))

    gitignore_added = ensure_gitignore(repo, [".tmp/", "**/.tmp/"])

    if args.with_templates:
        src = assets_dir / "AGENTS.template.md"
        existing_agents = find_root_agents(repo)
        dst = existing_agents or repo / "AGENTS.md"
        if src.exists() and (existing_agents is None or args.force):
            shutil.copyfile(src, dst)
            created.append(dst.name)

    holdout_created = []
    if args.init_holdout:
        holdout = Path(args.init_holdout).resolve()
        for sub in ["scenarios", "digital-twins", "seeds"]:
            target = holdout / sub
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                holdout_created.append(str(target))

    print(json.dumps({
        "status": "ok",
        "repo": str(repo),
        "created": created,
        "gitignore_added": gitignore_added,
        "holdout_created": holdout_created,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
