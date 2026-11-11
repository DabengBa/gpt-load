#!/usr/bin/env python3
"""Synchronize the lm-detector SDK and embedded data from one upstream commit."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

MODULE = "github.com/Ikaleio/lm-detector/sdk/go"
ROOT = Path(__file__).resolve().parent.parent
PIN = r"Data and SDK are pinned to upstream commit ([0-9a-f]{40})\."


def command(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise ValueError(f"expected exactly one match for {pattern}, found {count}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--upstream", default="https://github.com/Ikaleio/lm-detector.git")
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = repo / "internal/control/modeltrace_detector.go"
    notices = repo / "THIRD_PARTY_NOTICES.md"
    source_text = source.read_text()
    notice_text = notices.read_text()
    match = re.search(PIN, source_text)
    if not match:
        raise ValueError("missing upstream commit in modeltrace_detector.go")
    old = match.group(1)
    if not re.search(r"Embedded data snapshot: `" + old + r"`", notice_text):
        raise ValueError("upstream commit in third-party notice does not match")
    with tempfile.TemporaryDirectory(prefix="lm-detector-update-") as tmp:
        command("git", "clone", "--quiet", "--no-checkout", args.upstream, tmp)
        new = command("git", "rev-parse", "HEAD^{commit}", cwd=tmp)
        command("git", "cat-file", "-e", old + "^{commit}", cwd=tmp)
        if command("git", "diff", "--name-only", old, new, "--", "LICENSE", cwd=tmp):
            raise ValueError("upstream LICENSE changed; review it before updating")
        relevant = command("git", "diff", "--name-only", old, new, "--", "sdk/go", "data/unified_bank.json", "data/shared_detector.bin", cwd=tmp)
        if not relevant:
            print("No SDK or fingerprint data changes.")
            return
        data = {}
        for filename in ("unified_bank.json", "shared_detector.bin"):
            data[filename] = subprocess.run(
                ["git", "show", f"{new}:data/{filename}"], cwd=tmp, check=True, stdout=subprocess.PIPE
            ).stdout
            if not data[filename]:
                raise ValueError(f"upstream data/{filename} is empty")
        # Check metadata patterns before invoking Go, which modifies go.mod and go.sum.
        updated_source = replace_once(source_text, PIN, f"Data and SDK are pinned to upstream commit {new}.")
        notice_pattern = r"(## lm-detector\n(?:(?!\n## )[\s\S])*?- Go SDK: `github\.com/Ikaleio/lm-detector/sdk/go` version\n  `)[^`]+(`)"
        replace_once(notice_text, notice_pattern, r"\g<1>VERSION\g<2>")
        command("go", "get", f"{MODULE}@{new}", cwd=repo)
        command("go", "mod", "tidy", cwd=repo)
        version = command("go", "list", "-m", "-f", "{{.Version}}", MODULE, cwd=repo)
        updated_notice = replace_once(notice_text, notice_pattern, lambda m: m.group(1) + version + m.group(2))
        updated_notice = replace_once(updated_notice, r"(Embedded data snapshot: `)" + old + r"(`)", r"\g<1>" + new + r"\g<2>")
        for filename, contents in data.items():
            (repo / "internal/control/lmdata" / filename).write_bytes(contents)
        source.write_text(updated_source)
        notices.write_text(updated_notice)
        print(f"Updated lm-detector {old} -> {new} ({version})")


if __name__ == "__main__":
    main()
