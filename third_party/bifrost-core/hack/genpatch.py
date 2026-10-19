#!/usr/bin/env python3
"""Regenerate gptload-observer.patch from a pristine Bifrost core tree.

Usage: genpatch.py [--keep-whitespace] <pristine-tree> <fork-tree> <output-patch>

--keep-whitespace keeps trailing whitespace on added/removed lines, keeping only
the context-free format. It exists to demonstrate the diagnostics the default
mode removes; the artifact shipped in this repository uses the default.

The fork is a vendored copy of a released upstream module, so the patch is the
complete difference, not a code review diff. Two properties are required:

* the patch must apply to the pristine tree with `patch -p1` and reproduce the
  fork byte for byte;
* `git diff --check` must not report whitespace problems in the artifact.

Plain `diff -ru` emits three lines of context per hunk. A context line starts
with a single space, so a context line that lives inside a tab-indented block
(any Go block) begins " <TAB>..." and is reported as `space before tab in
indent`; a context line whose original text ends in whitespace is reported as
`trailing whitespace`. Both reports describe upstream source, not the fork.

Zero context lines (-U0) removes the ambiguity: every line of the artifact is
either a header, an added line (+) or a removed line (-), so the only possible
diagnostics are the fork's own added lines. `patch -p1` accepts the result and
reproduces the same tree; this generator verifies that itself before it writes
the final file.

The unified-context option must follow `-r` (GNU diff resets it to the default
otherwise) and must not be merged into the -ruN cluster. GNU diff also runs in
the C locale here: a localized build prints the no-newline marker translated, and
`patch` rejects that.

The artifact is applied with `patch -p1 -l`: whitespace in a hunk is ignored when
matching, so a removed line whose only difference from the pristine line is
trailing whitespace still applies against the tightened artifact.
"""

import os
import subprocess
import sys
import tempfile


def diff_env():
    """Run diff in the C locale so its output is the canonical English form.

    A localized diff translates the no-newline marker, and `patch` rejects the
    result.
    """
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def main() -> int:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    keep_whitespace = "--keep-whitespace" in sys.argv
    if len(args) != 3:
        print(__doc__.strip().splitlines()[4], file=sys.stderr)
        return 2
    pristine, fork, output = (os.path.abspath(p) for p in args)

    # diff exits 1 when the trees differ, which is the normal case. The output
    # is not parsed here: -U0 keeps every body line unambiguous, and rewriting
    # the headers with a parser risks mangling a body line that starts "--- ".
    diff = subprocess.run(
        ["diff", "-r", "-N", "-U0",
         "--exclude=.git", "--exclude=UPSTREAM.md",
         "--exclude=gptload-observer.patch", "--exclude=hack", pristine, fork],
        text=True, capture_output=True, env=diff_env(),
    )
    if diff.returncode not in (0, 1):
        print(diff.stderr, file=sys.stderr)
        return 1
    lines = []
    for line in diff.stdout.splitlines():
        if line.startswith(("diff -", "Only in ", "Binary files ")):
            continue
        if line.startswith("--- "):
            lines.append("--- a/" + os.path.relpath(line[4:].split("\t")[0],
                                                  pristine))
        elif line.startswith("+++ "):
            lines.append("+++ b/" + os.path.relpath(line[4:].split("\t")[0],
                                                    fork))
        else:
            lines.append(line)
    # Lines reaching this point are verbatim source, so a "-" line keeps the
    # whitespace it removes; stripping it would corrupt the patch. Lines the
    # fork adds are different: they are entirely under our control, so trailing
    # whitespace there is stripped (TAB-only lines become empty). Every other
    # line is a header or a hunk range with no trailing whitespace, so this also
    # keeps the artifact free of `git diff --check` diagnostics.
    if not keep_whitespace:
        for index, line in enumerate(lines):
            if line.startswith(("+", "-")) and not line.startswith(
                    ("+++ ", "--- ")):
                lines[index] = line.rstrip()
    payload = "\n".join(lines)
    # No trailing blank line: git reports one as "new blank line at EOF".
    payload = payload.rstrip("\n") + "\n"

    with tempfile.TemporaryDirectory() as scratch:
        copy = os.path.join(scratch, "tree")
        run(["cp", "-a", pristine, copy])
        run(["chmod", "-R", "u+w", copy])
        patch = os.path.join(scratch, "candidate.patch")
        with open(patch, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        if os.environ.get("GENPATCH_DEBUG"):
            print("pristine=%r fork=%r copy=%r" % (pristine, fork, copy),
                  file=sys.stderr)
        # -l ignores whitespace when matching, which keeps a removed line whose
        # only difference from the pristine line is trailing whitespace.
        run(["patch", "-p1", "--batch", "-l", "-i", patch], cwd=copy,
            stdout=subprocess.DEVNULL)
        replay = subprocess.run(
            ["diff", "-rq", "--exclude=.git", "--exclude=UPSTREAM.md",
             "--exclude=gptload-observer.patch", "--exclude=hack",
             "--exclude=*.orig", "--exclude=*.rej", copy, fork],
            text=True, capture_output=True,
        )
        if replay.stdout.strip():
            print("replay does not reproduce the fork:\n" + replay.stdout,
                  file=sys.stderr)
            return 1

    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)

    print("%s: %d bytes, %d files, replay reproduces the fork"
          % (output, len(payload),
             sum(1 for line in lines if line.startswith("--- a/"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
