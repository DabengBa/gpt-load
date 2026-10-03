#!/usr/bin/env python3
"""Run the remote activation script without Docker or production files."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


MOCK = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "docker":
    if args[:2] == ["volume", "inspect"]:
        print(os.environ["DATA_DIR"])
    elif args and args[0] == "inspect":
        if any("Mount" in arg for arg in args):
            print(os.environ["DATA_DIR"])
        elif any("State.Running" in arg for arg in args):
            print("false" if Path(os.environ["RUNNING_STATE"]).read_text() == "stop" else "true")
        else:
            print("healthy")
    elif args and args[0] == "compose" and "stop" in args:
        Path(os.environ["RUNNING_STATE"]).write_text("stop")
    elif args and args[0] in {"stop", "start"}:
        Path(os.environ["RUNNING_STATE"]).write_text(args[0])
    sys.exit(0)
if name == "tar" and os.environ["FAIL_BACKUP"] == "1":
    # Fail creation only: listing an archive is not a backup failure.
    if any((arg.startswith("-") and "c" in arg) or
           (not arg.startswith("-") and arg.startswith("c")) or
           arg == "--create" for arg in args):
        print("injected backup creation failure", file=sys.stderr)
        sys.exit(73)
if name == "du":
    print("1\t" + os.environ["DATA_DIR"])
elif name == "df":
    print("Avail")
    print("999999998")
elif name == "findmnt":
    print("/")
elif name == "curl":
    print('{"status":"ok"}')
elif name == "sleep":
    pass
else:
    real = json.loads(os.environ["REAL_COMMANDS"])[name]
    sys.exit(subprocess.run([real, *args]).returncode)
'''


class RemoteActivationTests(unittest.TestCase):
    def run_activation(self, fail_backup=False):
        source = Path(__file__).with_name("deploy.sh").read_text()
        lines = source.splitlines()
        starts = [i for i, line in enumerate(lines) if "<<'REMOTE'" in line]
        self.assertEqual(len(starts), 1, "expected one REMOTE heredoc")
        start = starts[0] + 1
        end = lines.index("REMOTE", start)
        remote = "\n".join(lines[start:end]) + "\n"
        self.assertEqual(remote.count("cd /opt/gpt-load"), 1)

        with tempfile.TemporaryDirectory(prefix="deploy-activation-") as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "gpt-load.db").write_text("old database fixture\n")
            compose = root / "docker-compose.yml"
            original = "services:\n  gpt-load:\n    image: gpt-load:old\n"
            compose.write_text(original)
            commands = root / "commands.jsonl"
            commands.touch()
            state = root / "running"
            state.write_text("running")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            mock = bin_dir / "mock"
            mock.write_text(MOCK)
            mock.chmod(0o755)
            names = ("docker", "cp", "tar", "sed", "du", "df", "findmnt",
                     "date", "mktemp", "curl", "sleep")
            real = {name: shutil.which(name) for name in names}
            for name in names:
                (bin_dir / name).symlink_to(mock)
            environment = {
                **os.environ,
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "TMPDIR": str(root),
                "COMMAND_LOG": str(commands),
                "DATA_DIR": str(data),
                "RUNNING_STATE": str(state),
                "REAL_COMMANDS": json.dumps(real),
                "FAIL_BACKUP": "1" if fail_backup else "0",
            }
            remote = remote.replace("cd /opt/gpt-load", "cd " + shlex.quote(str(root)))
            result = subprocess.run(
                ["bash", "-s", "--", "gpt-load:new"], input=remote,
                text=True, capture_output=True, env=environment,
                cwd=root, timeout=10,
            )
            events = [json.loads(line) for line in commands.read_text().splitlines()]
            return result, events, compose.read_text(), original, state.read_text()

    def event_index(self, events, predicate, label):
        matches = [i for i, event in enumerate(events) if predicate(event)]
        self.assertTrue(matches, f"missing {label}: {events}")
        return matches[0]

    @staticmethod
    def is_archive(event, operation):
        return event[0] == "tar" and any(
            arg == {"c": "--create", "t": "--list"}[operation]
            or (arg.startswith("-") and not arg.startswith("--") and operation in arg)
            or (not arg.startswith("-") and arg.startswith(operation))
            for arg in event[1:]
        )

    @staticmethod
    def is_switch(event):
        return event[0] == "sed" and any(arg.startswith("-i") for arg in event[1:])

    @staticmethod
    def is_up(event):
        return event[:2] == ["docker", "compose"] and "up" in event

    @staticmethod
    def is_stop(event):
        return event[0] == "docker" and "gpt-load" in event and (
            event[1] == "stop" or (event[1] == "compose" and "stop" in event)
        )

    def test_success_stops_and_validates_backup_before_switch(self):
        result, events, compose, _, _ = self.run_activation()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        precheck = self.event_index(events, lambda e: e[:3] == ["docker", "image", "inspect"], "image precheck")
        space = self.event_index(events, lambda e: e[0] == "df", "free-space precheck")
        stop = self.event_index(events, self.is_stop, "old container stop")
        backup = self.event_index(events, lambda e: self.is_archive(e, "c"), "backup creation")
        validate = self.event_index(events, lambda e: self.is_archive(e, "t"), "backup validation")
        switch = self.event_index(events, self.is_switch, "compose image switch")
        up = self.event_index(events, self.is_up, "new container start")
        self.assertLess(precheck, stop)
        self.assertLess(space, stop)
        self.assertLess(stop, backup)
        self.assertLess(backup, validate)
        self.assertLess(validate, switch)
        self.assertLess(switch, up)
        self.assertIn("image: gpt-load:new", compose)
        self.assertFalse(any(e[:2] == ["docker", "start"] for e in events))

    def test_backup_failure_restarts_old_without_switching(self):
        result, events, compose, original, state = self.run_activation(fail_backup=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("injected backup creation failure", result.stderr)
        stop = self.event_index(events, self.is_stop, "old container stop")
        backup = self.event_index(events, lambda e: self.is_archive(e, "c"), "failed backup")
        restart = self.event_index(events, lambda e: e[:2] == ["docker", "start"] and "gpt-load" in e, "old container recovery")
        self.assertLess(stop, backup)
        self.assertLess(backup, restart)
        self.assertEqual(state, "start")
        self.assertEqual(compose, original)
        self.assertFalse(any(self.is_switch(e) for e in events), events)
        self.assertFalse(any(self.is_up(e) for e in events), events)


if __name__ == "__main__":
    unittest.main()
