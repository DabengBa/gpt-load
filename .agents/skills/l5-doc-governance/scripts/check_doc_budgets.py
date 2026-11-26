#!/usr/bin/env python3
"""Validate L5 documentation budgets, schemas, .tmp/ hygiene, and dangling symbols."""
import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

# Same durable-owner roots indexed by prepare_distill_bundle.existing_owner_files().
OWNER_DIRS = (".docs/adr", ".docs/tech", ".docs/db")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end_idx is None:
        return {}, text
    data: dict = {}
    current_key = None
    multiline_scalar = False
    for raw in lines[1:end_idx]:
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if not raw.startswith(" ") and re.match(r"^[A-Za-z0-9_-]+:\s*", raw):
            k, v = raw.split(":", 1)
            k, v = k.strip(), v.strip()
            current_key = k
            multiline_scalar = False
            if v.startswith("[") and v.endswith("]"):
                data[k] = [x.strip().strip("\"'") for x in v[1:-1].split(",") if x.strip()]
            elif v in (">-", ">", "|", "|-"):
                data[k] = ""
                multiline_scalar = True
            elif v == "":
                data[k] = {}
            else:
                data[k] = v.strip("\"'")
        elif raw.lstrip().startswith("- ") and current_key:
            item = raw.lstrip()[2:].strip().strip("\"'")
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(item)
        elif raw.startswith("  ") and current_key:
            if multiline_scalar:
                data[current_key] = (str(data.get(current_key, "")) + " " + raw.strip()).strip()
            elif isinstance(data.get(current_key), dict) and ":" in raw:
                subk, subv = raw.strip().split(":", 1)
                data[current_key][subk.strip()] = subv.strip().strip("\"'")
    return data, "\n".join(lines[end_idx + 1:])


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(0, len(text) - cjk)
    return int(cjk / 1.4 + other / 3.8)


def find_root_agents(repo: Path) -> Path | None:
    return next(
        (p for p in sorted(repo.glob("*")) if p.is_file() and p.name.lower() == "agents.md"),
        None,
    )


def check_git_tmp_hygiene(repo: Path) -> list[str]:
    issues: list[str] = []
    if not (repo / ".git").exists():
        return issues
    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", ":(glob).tmp/**", ":(glob)**/.tmp/**"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if tracked:
            issues.append(f"Tracked files found under .tmp/: {tracked.splitlines()}")
        for probe in (".tmp/probe", "packages/sub/.tmp/probe"):
            rc = subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            if rc != 0:
                issues.append(f"'{probe}' is NOT ignored by git; ephemeral process docs risk leaking.")
    except FileNotFoundError:
        pass
    return issues


def validate_skill_dir(skill_path: Path, skill_max_lines: int) -> list[str]:
    errs: list[str] = []
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return [f"Skill directory missing SKILL.md: {skill_path.name}"]
    txt = skill_md.read_text(encoding="utf-8")
    lines = len(txt.splitlines())
    if lines > skill_max_lines:
        errs.append(f"{skill_path.name}/SKILL.md exceeds {skill_max_lines} lines ({lines}).")
    fm, _ = parse_frontmatter(txt)
    name = fm.get("name", "")
    desc = fm.get("description", "")
    if name != skill_path.name:
        errs.append(f"{skill_path.name}/SKILL.md name '{name}' does not match directory name.")
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", str(name)) or len(str(name)) > 64:
        errs.append(f"Invalid skill name '{name}' per agentskills.io spec.")
    if not isinstance(desc, str) or not desc.strip() or len(desc) > 1024:
        errs.append(f"{skill_path.name}/SKILL.md invalid description length ({len(str(desc))}).")
    return errs


def check_affects_symbols(repo: Path, doc: Path, syms) -> list[str]:
    """Dangling Symbol GC for decisions in .knowledge/ or mapped durable owners."""
    errs: list[str] = []
    if isinstance(syms, str):
        syms = [syms]
    rel = doc.relative_to(repo)
    for sym in syms:
        file_part, _, symbol_part = sym.partition("#")
        target = repo / file_part
        if not target.exists():
            errs.append(f"Dangling decision ref in {rel}: file '{file_part}' not found.")
        elif symbol_part and symbol_part not in target.read_text(encoding="utf-8", errors="ignore"):
            errs.append(f"Dangling symbol '{symbol_part}' in {file_part} referenced by {rel}.")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate L5 doc budgets, schemas, and references.")
    parser.add_argument("--repo", default=".", help="Repository root path.")
    parser.add_argument("--agents-max-lines", type=int, default=120)
    parser.add_argument("--agents-max-tokens", type=int, default=1500)
    parser.add_argument("--skill-max-lines", type=int, default=500)
    parser.add_argument("--check-skill", default="", help="Optional path to a standalone skill directory to validate.")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    if args.check_skill:
        errors.extend(validate_skill_dir(Path(args.check_skill).resolve(), args.skill_max_lines))
        print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
        return 0 if not errors else 1

    repo = Path(args.repo).resolve()
    root_agents = find_root_agents(repo)
    budgets: dict = {}
    if root_agents:
        txt = root_agents.read_text(encoding="utf-8")
        lines, toks = len(txt.splitlines()), estimate_tokens(txt)
        budgets["root_agents"] = {
            "path": root_agents.name, "lines": lines, "max_lines": args.agents_max_lines,
            "est_tokens": toks, "max_tokens": args.agents_max_tokens,
        }
        if lines > args.agents_max_lines:
            errors.append(f"AGENTS.md exceeds line budget: {lines} > {args.agents_max_lines}")
        if toks > args.agents_max_tokens:
            errors.append(f"AGENTS.md exceeds estimated token budget: {toks} > {args.agents_max_tokens}")
    else:
        warnings.append("Root AGENTS.md not found.")

    skills_dir = repo / ".agents" / "skills"
    skills_checked = 0
    if skills_dir.exists():
        for skill_path in sorted(skills_dir.iterdir()):
            if skill_path.is_dir() and not skill_path.name.startswith(("_", ".")):
                skills_checked += 1
                errors.extend(validate_skill_dir(skill_path, args.skill_max_lines))

    knowledge = repo / ".knowledge"
    counts = {"skills": skills_checked, "decisions": 0, "invariants": 0, "quirks": 0}

    def docs(category: str) -> list[Path]:
        d = knowledge / category
        return sorted(d.glob("*.md")) if d.exists() else []

    for f in docs("decisions"):
        counts["decisions"] += 1
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        for req in ("id", "status", "affects_symbols"):
            if req not in fm:
                errors.append(f"{f.relative_to(repo)} missing frontmatter field '{req}'.")
        if fm.get("status") == "active":
            errors.extend(check_affects_symbols(repo, f, fm.get("affects_symbols", [])))

    # Mapped ADR owners keep L5 `affects_symbols` in their own frontmatter; the
    # repo's own docs:check entry does not validate symbol references.
    for rel in OWNER_DIRS:
        owner_dir = repo / rel
        if not owner_dir.is_dir():
            continue
        for f in sorted(owner_dir.rglob("*.md")):
            fm, _ = parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
            if not fm.get("affects_symbols") or fm.get("status", "active") != "active":
                continue
            errors.extend(check_affects_symbols(repo, f, fm["affects_symbols"]))

    for f in docs("invariants"):
        counts["invariants"] += 1
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        if "id" not in fm or not fm.get("paths"):
            errors.append(f"{f.relative_to(repo)} missing 'id' or non-empty 'paths'.")
        else:
            patterns = fm["paths"] if isinstance(fm["paths"], list) else [fm["paths"]]
            if not any(glob.glob(str(repo / pat), recursive=True) for pat in patterns):
                warnings.append(f"Orphaned invariant paths in {f.name}: {patterns} match 0 files.")

    for f in docs("quirks"):
        counts["quirks"] += 1
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        if "id" not in fm or "external_service" not in fm:
            errors.append(f"{f.relative_to(repo)} missing 'id' or 'external_service'.")

    errors.extend(check_git_tmp_hygiene(repo))

    print(json.dumps({
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "budgets": budgets,
        "counts": counts,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
