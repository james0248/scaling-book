#!/usr/bin/env python3
"""
TPU VM via the queued-resources API, wired up as a JupyterLab / Colab runtime.
Queued resources block until capacity appears instead of failing immediately,
which matters because v6e spot capacity is frequently dry.

    ./tpu.py            choose accel/pricing/zone -> queue -> poll -> jupyter -> URL
    ./tpu.py url        reopen the tunnel if needed and print the URL
    ./tpu.py status     what is running, queue state, tunnel state
    ./tpu.py ssh        shell on the VM
    ./tpu.py delete     tear down (stops billing)

The tunnel runs detached, so the script returns instead of blocking.
ACCEL, SPOT and ZONE env vars each skip their selection stage:
    SPOT=0 ZONE=us-west1-c ./tpu.py     on-demand, no prompts
"""

import json
import os
import secrets
import signal
import subprocess
import sys
import termios
import time
import tty
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = os.environ.get("PROJECT_ID", "visionary-491008")
TPU_NAME = os.environ.get("TPU_NAME", "jax8")
QR_ID = os.environ.get("QR_ID", f"{TPU_NAME}-qr")
ACCEL_ENV = os.environ.get("ACCEL")
SPOT_ENV = os.environ.get("SPOT")
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", 8888))
TB_PORT = int(os.environ.get("TB_PORT", 6006))  # %tensorboard iframes localhost:6006
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", 20))

# v6e-8 is the ONLY 8-device option on this project: v5litepod-8 is rejected by
# TPUV5sPreemptibleLitepodServingPerProjectPerZone (limit 4, every zone) via
# both `tpu-vm create` and `queued-resources create`, so v5e tops out at 4.
#
# Zones listed are those where this project has nonzero *spot training* quota
# AND the chip is offered. On-demand quota is a separate (usually zero) pool, so
# an on-demand request may be rejected in every zone listed here.
ACCELERATORS = {
    "v6e-8": (
        "us-east5-b us-east5-a us-central1-a us-central1-b us-south1-a us-west1-c asia-northeast1-b".split(),
        "v2-alpha-tpuv6e",
    ),
    "v5litepod-4": (
        "us-west1-c us-south1-a us-west4-a europe-west4-a asia-northeast1-b".split(),
        "v2-alpha-tpuv5-lite",
    ),
}

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / ".tpu-state.json"
ZONE_FILE = HERE / ".tpu-zone"  # superseded by STATE_FILE; still read once, to migrate
TOKEN_FILE = HERE / ".tpu-token"
PID_FILE = HERE / ".tpu-tunnel.pid"
TUNNEL_LOG = HERE / ".tpu-tunnel.log"
SSH_KEY = Path.home() / ".ssh" / "google_compute_engine"

INSTALL = """
set -e
export PATH="$HOME/.local/bin:$PATH"   # pip lands jupyter here, not on default PATH
pipi() { python3 -m pip install -q "$@" || python3 -m pip install -q --break-system-packages "$@"; }
# jupyterlab: the browser UI. jupyter_http_over_ws: only Colab's local runtime
# needs it. tensorboard: NOT a dependency of the profile plugin, and %load_ext
# tensorboard fails without it (Colab preinstalls it, a plain VM does not).
pipi -U 'jax[tpu]' jupyterlab ipykernel jupyter_http_over_ws tensorboard tensorboard-plugin-profile
jupyter server extension enable --py jupyter_http_over_ws --user || true
python3 -c "
import jax
d = jax.devices()
print(f'jax {jax.__version__}: {len(d)} devices -> {d[0].device_kind}')
assert len(d) == __DEVICES__, f'expected __DEVICES__ devices, got {len(d)}'
"
"""

START_JUPYTER = """
export PATH="$HOME/.local/bin:$PATH"
# bracket the pattern, else pkill matches this very command line and kills its
# own ssh session (exit 255).
pkill -f '[j]upyter-server' 2>/dev/null || true
sleep 1
nohup jupyter server \
  --ServerApp.allow_origin='https://colab.research.google.com' \
  --ServerApp.token='__TOKEN__' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.allow_remote_access=True \
  --ServerApp.port_retries=0 \
  --port=8888 --no-browser \
  > ~/jupyter.log 2>&1 < /dev/null &
sleep 7
curl -sf -m 5 http://localhost:8888/api >/dev/null \
  || { echo '--- jupyter failed ---'; tail -20 ~/jupyter.log; exit 1; }
echo 'jupyter server up'
"""


def die(msg):
    sys.exit(f"ERROR: {msg}")


ENV = {**os.environ, "CLOUDSDK_CORE_PROJECT": PROJECT}


def gcloud(*args, capture=False, check=True, quiet=False):
    run = subprocess.run(
        ["gcloud", *args],
        env=ENV,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if capture or quiet else None,
    )
    return run.stdout.strip() if capture else run.returncode


def ssh_args(zone, *extra):
    return [
        "compute", "tpus", "tpu-vm", "ssh", TPU_NAME,
        f"--zone={zone}", "--worker=0", "--quiet", *extra,
    ]


def ssh_vm(zone, command, **kw):
    return gcloud(*ssh_args(zone, f"--command={command}"), **kw)


def qr_state(zone):
    return gcloud(
        "compute", "tpus", "queued-resources", "describe", QR_ID,
        f"--zone={zone}", "--format=value(state.state)",
        capture=True, check=False,
    )


def pick(title, options, default=0):
    """Arrow-key menu. Returns the chosen option; falls back to the default with
    no tty (CI, piped)."""
    try:
        tty_io = open("/dev/tty", "r+b", buffering=0)
    except OSError:
        return options[default]

    def draw(s):
        tty_io.write(s.encode())

    fd, sel = tty_io.fileno(), default
    saved = termios.tcgetattr(fd)
    draw(f"\n  {title}  [up/down, enter]\n\n")
    try:
        tty.setraw(fd)
        while True:
            for i, opt in enumerate(options):
                draw(f"  \033[36m> {opt}\033[0m\r\n" if i == sel else f"    {opt}\r\n")
            key = tty_io.read(1)
            if key == b"\x1b":  # arrow: ESC [ A/B, one byte at a time
                key = {b"[A": b"k", b"[B": b"j"}.get(tty_io.read(1) + tty_io.read(1), b"")
            if key in (b"\r", b"\n", b""):
                break
            if key == b"\x03":  # raw mode swallows SIGINT
                raise KeyboardInterrupt
            if key == b"k":
                sel = max(0, sel - 1)
            elif key == b"j":
                sel = min(len(options) - 1, sel + 1)
            draw(f"\033[{len(options)}A")  # rewind cursor to redraw
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    draw("\n")
    return options[sel]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    if ZONE_FILE.exists():  # migrate a pod queued by an older version
        return {"zone": ZONE_FILE.read_text().strip(), "accel": "v6e-8", "spot": True}
    return {}


def save_state(zone, accel, spot):
    STATE_FILE.write_text(json.dumps({"zone": zone, "accel": accel, "spot": spot}, indent=2))
    ZONE_FILE.unlink(missing_ok=True)


def index_of(value, options, fallback=0):
    return options.index(value) if value in options else fallback


def ensure_ssh_key():
    """The detached tunnel has no tty, so a passphrase-protected key must already
    be in ssh-agent. Answering ssh's prompt interactively does not cache it."""
    if not SSH_KEY.exists():
        return
    no_passphrase = subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", str(SSH_KEY)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if no_passphrase:
        return
    fingerprint = subprocess.run(
        ["ssh-keygen", "-lf", str(SSH_KEY)], capture_output=True, text=True
    ).stdout.split()
    loaded = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True).stdout
    if len(fingerprint) > 1 and fingerprint[1] in loaded:
        return
    print(">> adding ssh key to agent (passphrase asked once) ...")
    keychain = ["--apple-use-keychain"] if sys.platform == "darwin" else []
    if subprocess.run(["ssh-add", *keychain, str(SSH_KEY)]).returncode != 0:
        die(f"could not add {SSH_KEY} to ssh-agent")


def jupyter_reachable():
    try:
        urllib.request.urlopen(f"http://localhost:{LOCAL_PORT}/api", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def stop_tunnel():
    if not PID_FILE.exists():
        return
    try:
        # gcloud runs ssh as a subprocess, so killing its pid orphans the
        # tunnel. start_new_session made the pid a process-group leader.
        os.killpg(int(PID_FILE.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
    PID_FILE.unlink()


def start_tunnel(zone):
    stop_tunnel()
    ensure_ssh_key()
    with open(TUNNEL_LOG, "w") as log:
        proc = subprocess.Popen(
            ["gcloud", *ssh_args(zone), "--", "-N",
             "-L", f"{LOCAL_PORT}:localhost:8888",
             "-L", f"{TB_PORT}:localhost:6006"],
            env=ENV,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    PID_FILE.write_text(str(proc.pid))
    for _ in range(30):
        if jupyter_reachable():
            return
        if proc.poll() is not None:
            break
        time.sleep(2)
    stop_tunnel()
    die(f"tunnel never came up on port {LOCAL_PORT}:\n{TUNNEL_LOG.read_text().strip()}")


def print_url(zone, accel, spot):
    devices = accel.rsplit("-", 1)[1]
    token = TOKEN_FILE.read_text().strip()
    print(f"""
============================================================
  {accel} ({"spot" if spot else "on-demand"}) in {zone} -- {devices} JAX devices.

  JupyterLab in a browser:
    http://localhost:{LOCAL_PORT}/lab?token={token}

  Or paste into Colab (Connect -> "Connect to a local runtime"):
    http://localhost:{LOCAL_PORT}/?token={token}

  TensorBoard on :{TB_PORT} is forwarded too, so %tensorboard works.

  The tunnel runs in the background.
  Tear down when done:  ./tpu.py delete
============================================================
""")


def saved_config():
    st = load_state()
    zone = os.environ.get("ZONE") or st.get("zone")
    if not zone:
        die("nothing recorded. Run ./tpu.py first.")
    return zone, st.get("accel", "v6e-8"), st.get("spot", True)


def require_active(zone):
    state = qr_state(zone)
    if state != "ACTIVE":
        die(f"TPU not ACTIVE (state: {state or 'not found'}).")


def delete_qr(zone):
    # The API refuses deletion in transitional states ("not supported when state
    # is PROVISIONING; must be one of [ACCEPTED WAITING_FOR_RESOURCES SUSPENDED
    # FAILED]"). The rejected request still nudges PROVISIONING -> SUSPENDING ->
    # FAILED, which is deletable, so keep retrying until it takes.
    for _ in range(60):
        state = qr_state(zone)
        if not state:
            return
        if gcloud("compute", "tpus", "queued-resources", "delete", QR_ID,
                  f"--zone={zone}", "--force", "--quiet", check=False, quiet=True) == 0:
            return
        print(f"   {state} (not deletable yet) ...")
        time.sleep(10)
    die(f"could not delete {QR_ID} in {zone}; it is stuck in {qr_state(zone)}.")


def wait_for_capacity(zone):
    # Safe to Ctrl-C while WAITING_FOR_RESOURCES: the request keeps queuing
    # server-side. Re-run to resume waiting.
    print(">> waiting for capacity (Ctrl-C is safe; the queue keeps running) ...")
    while True:
        state = qr_state(zone)
        if state == "ACTIVE":
            print("   ACTIVE")
            return
        if state in ("FAILED", "SUSPENDED"):
            gcloud("compute", "tpus", "queued-resources", "describe", QR_ID, f"--zone={zone}", check=False)
            die(f"queued resource ended in state {state}")
        print(f"   {state or '(not found yet)'} ...")
        time.sleep(POLL_SECONDS)


def wait_for_ssh(zone):
    # A fresh VM takes ~1-2 min to sync SSH keys from project metadata; until
    # then every connection is "Permission denied (publickey)".
    print(">> waiting for ssh ...")
    for _ in range(10):
        if ssh_vm(zone, "true", check=False, quiet=True) == 0:
            return
        time.sleep(20)
    die("ssh never came up.")


SPOT_LABELS = ["spot (preemptible, reclaimed without warning)",
               "on-demand (not preempted, needs separate quota)"]


def select_config():
    """Stages: accelerator -> pricing -> zone -> confirm. Each stage defaults to
    the previous run's choice; each is skipped when its env var is set."""
    prev = load_state()
    accels = list(ACCELERATORS)

    accel = ACCEL_ENV or pick("1/4  Accelerator", accels, index_of(prev.get("accel"), accels))
    if accel not in ACCELERATORS:
        die(f"unsupported ACCEL '{accel}' (choose one of {', '.join(accels)})")

    if SPOT_ENV is None:
        spot = pick("2/4  Pricing", SPOT_LABELS, 0 if prev.get("spot", True) else 1) == SPOT_LABELS[0]
    else:
        spot = SPOT_ENV not in ("0", "false", "no")

    zones = ACCELERATORS[accel][0]
    zone = os.environ.get("ZONE") or pick("3/4  Zone", zones, index_of(prev.get("zone"), zones))

    kind = "spot" if spot else "on-demand"
    if pick(f"4/4  Queue {accel} ({kind}) in {zone}?", ["yes, submit", "no, cancel"]) != "yes, submit":
        sys.exit("cancelled.")
    return accel, spot, zone


def up():
    prev = load_state()
    accel, spot, zone = select_config()
    zones, runtime = ACCELERATORS[accel]
    kind = "spot" if spot else "on-demand"

    # Switching zones would strand the old queued resource, which keeps billing.
    last = prev.get("zone")
    if last and last != zone and qr_state(last):
        die(f"a queued resource still exists in {last}. Run ./tpu.py delete first.")

    state = qr_state(zone)
    if state and prev and (prev.get("accel"), prev.get("spot")) != (accel, spot):
        die(f"{QR_ID} already exists in {zone} as {prev.get('accel')} "
            f"({'spot' if prev.get('spot') else 'on-demand'}). "
            f"Run ./tpu.py delete to requeue it as {accel} ({kind}).")
    save_state(zone, accel, spot)

    if not state:
        print(f">> queuing {accel} ({kind}) in {zone} ...")
        gcloud(
            "compute", "tpus", "queued-resources", "create", QR_ID,
            f"--node-id={TPU_NAME}", f"--zone={zone}",
            f"--accelerator-type={accel}", f"--runtime-version={runtime}",
            *(["--spot"] if spot else []),
        )
    elif state in ("FAILED", "SUSPENDED"):
        gcloud("compute", "tpus", "queued-resources", "describe", QR_ID, f"--zone={zone}", check=False)
        die(f"queued resource is {state}. Clean up first: ./tpu.py delete")
    else:
        print(f">> reusing queued resource (state: {state})")

    wait_for_capacity(zone)
    ensure_ssh_key()
    wait_for_ssh(zone)

    print(">> installing jax + jupyter (first run takes a few minutes) ...")
    ssh_vm(zone, INSTALL.replace("__DEVICES__", accel.rsplit("-", 1)[1]))

    token = secrets.token_hex(24)
    TOKEN_FILE.write_text(token + "\n")
    TOKEN_FILE.chmod(0o600)
    print(">> starting jupyter server ...")
    ssh_vm(zone, START_JUPYTER.replace("__TOKEN__", token))

    print(f">> opening tunnel localhost:{LOCAL_PORT} -> VM:8888 ...")
    start_tunnel(zone)
    print_url(zone, accel, spot)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    if cmd == "up":
        up()
    elif cmd == "delete":
        zone, _, _ = saved_config()
        stop_tunnel()
        print(f">> deleting {QR_ID} in {zone} ...")
        delete_qr(zone)
        STATE_FILE.unlink(missing_ok=True)
        ZONE_FILE.unlink(missing_ok=True)
        TOKEN_FILE.unlink(missing_ok=True)
        print("Deleted. Billing stopped.")
    elif cmd == "status":
        zone, accel, spot = saved_config()
        print(f"tpu:    {accel} ({'spot' if spot else 'on-demand'})")
        print(f"zone:   {zone}")
        print(f"state:  {qr_state(zone) or 'not found'}")
        print(f"tunnel: {'up' if jupyter_reachable() else 'down'}")
    elif cmd == "ssh":
        zone, _, _ = saved_config()
        require_active(zone)
        os.execvpe("gcloud", ["gcloud", *ssh_args(zone)], ENV)
    elif cmd == "url":
        zone, accel, spot = saved_config()
        require_active(zone)
        if not TOKEN_FILE.exists():
            die("no saved token. Run ./tpu.py first.")
        if not jupyter_reachable():
            start_tunnel(zone)
        print_url(zone, accel, spot)
    else:
        sys.exit("usage: ./tpu.py [up|url|status|ssh|delete]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        die(f"{' '.join(e.cmd)} exited {e.returncode}")
