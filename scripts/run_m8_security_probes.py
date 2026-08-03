"""Execute an adversarial self-inspection inside the exact pinned M8 Docker sandbox."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from nanopt.agent.sandbox import DockerSandboxBackend, SandboxLimits
from nanopt.runtime.artifacts import write_json, write_text

PROBE = r"""import glob
import json
import os
import socket
from pathlib import Path

network_blocked = False
try:
    connection = socket.create_connection(("1.1.1.1", 53), timeout=0.5)
    connection.close()
except OSError:
    network_blocked = True

root_write_blocked = False
try:
    Path("/nanopt-security-probe").write_text("bad")
except OSError:
    root_write_blocked = True

workspace_write_ok = False
try:
    Path("workspace-write-probe.txt").write_text("ok")
    workspace_write_ok = True
except OSError:
    pass

status = Path("/proc/self/status").read_text()
fields = dict(
    line.split(":", 1) for line in status.splitlines() if ":" in line
)
print(json.dumps({
    "uid": os.getuid(),
    "gid": os.getgid(),
    "network_blocked": network_blocked,
    "root_write_blocked": root_write_blocked,
    "workspace_write_ok": workspace_write_ok,
    "gpu_devices": glob.glob("/dev/nvidia*"),
    "docker_socket_present": Path("/var/run/docker.sock").exists(),
    "cap_eff": fields.get("CapEff", "").strip(),
    "no_new_privs": fields.get("NoNewPrivs", "").strip(),
    "home": os.environ.get("HOME"),
}, sort_keys=True))
"""


def run_probes(image: str) -> dict[str, object]:
    backend = DockerSandboxBackend(image)
    inspection = backend.validate_available()
    with tempfile.TemporaryDirectory(prefix="nanopt-m8-security-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir(mode=0o777)
        workspace.chmod(0o777)
        probe = workspace / "probe.py"
        write_text(probe, PROBE)
        probe.chmod(0o644)
        execution = backend.run(
            ["python", "probe.py"],
            workspace,
            SandboxLimits(timeout_seconds=5, memory_mib=256, pids=32),
        )
    if execution.status != "completed" or execution.exit_code != 0:
        raise RuntimeError(f"Docker security probe failed: {execution.output}")
    try:
        result = json.loads(execution.output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Docker security probe returned malformed output") from exc
    expected = {
        "uid": 65532,
        "gid": 65532,
        "network_blocked": True,
        "root_write_blocked": True,
        "workspace_write_ok": True,
        "gpu_devices": [],
        "docker_socket_present": False,
        "cap_eff": "0000000000000000",
        "no_new_privs": "1",
        "home": "/tmp",
    }
    if result != expected:
        raise RuntimeError(f"Docker sandbox security contract differs: {result}")
    return {
        "schema_version": 1,
        "status": "passed",
        "image": image,
        "docker": inspection,
        "probe": result,
        "backend_details": execution.backend_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probes(args.image)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
