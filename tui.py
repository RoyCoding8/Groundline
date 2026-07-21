"""Distortion Engine — lightweight TUI launcher.

Usage:
    python tui.py
    uv run python tui.py
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("rich is not installed. Run: uv sync")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
EXAMPLE_ENV = ROOT / ".env.example"

API_PORT = 8000
FRONTEND_PORT = 5173

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _version() -> str:
    init = ROOT / "src" / "distortion_engine" / "__init__.py"
    for line in init.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip("\"'")
    return "?"


def _env_status() -> dict[str, str]:
    """Read .env and report key presence (never print values)."""
    keys = [
        "DISTORTION_MODEL",
        "DISTORTION_API_BASE",
        "DISTORTION_API_KEY",
        "DISTORTION_TIMEOUT_SECONDS",
        "DISTORTION_MAX_ATTEMPTS",
    ]
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    result: dict[str, str] = {}
    for k in keys:
        v = values.get(k, "")
        if not v:
            result[k] = "[dim]not set[/]"
        elif k == "DISTORTION_API_KEY":
            result[k] = "[green]set[/]"
        else:
            result[k] = f"[green]{v}[/]"
    return result


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command in a subprocess, forwarding stdio."""
    proc = subprocess.run(cmd, cwd=cwd or ROOT, shell=(sys.platform == "win32"))
    return proc.returncode


def _start_background(cmd: list[str], cwd: Path | None = None) -> subprocess.Popen:
    """Start a process in the background, returning the Popen object.

    Uses DEVNULL to prevent pipe-buffer deadlocks when the child produces
    more output than the OS pipe buffer can hold. On Unix, creates a new
    process group via os.setsid so the entire tree can be killed cleanly.
    """
    kwargs: dict = {
        "cwd": cwd or ROOT,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["shell"] = True
    else:
        kwargs["preexec_fn"] = os.setsid  # new process group for tree kill
    return subprocess.Popen(cmd, **kwargs)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and all its children across platforms."""
    if proc.poll() is not None:
        return  # already exited
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()


def _kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except OSError:
        return  # nothing on that port
    # Windows: find and kill the process
    if sys.platform == "win32":
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
        )
        for line in out.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.run(
                    ["taskkill", "/F", "/PID", pid],
                    capture_output=True,
                )
                console.print(f"  [yellow]Killed process on port {port} (PID {pid})[/]")
    else:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        for pid in out.stdout.strip().splitlines():
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
                console.print(f"  [yellow]Killed process on port {port} (PID {pid})[/]")


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll until a port is accepting connections."""
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------


def action_launch_web() -> None:
    """Start backend + frontend together, print URLs, wait for Ctrl+C."""
    console.print()
    console.print("[bold]Launching Distortion Engine…[/]\n")

    # Kill anything on our ports first
    _kill_port(API_PORT)
    _kill_port(FRONTEND_PORT)

    # Start backend
    backend = _start_background(
        [sys.executable, "-m", "uvicorn", "distortion_engine.api.app:app", "--port", str(API_PORT)],
    )
    console.print(f"  [cyan]Backend[/] starting on port {API_PORT}…")

    # Start frontend
    fe = ROOT / "frontend"
    if not fe.exists():
        console.print("  [red]frontend/ not found — skipping[/]")
        frontend = None
    else:
        frontend = _start_background(["npm", "run", "dev"], cwd=fe)
        console.print(f"  [cyan]Frontend[/] starting on port {FRONTEND_PORT}…")

    # Wait for both to be ready (in parallel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        api_future = pool.submit(_wait_for_port, "127.0.0.1", API_PORT)
        fe_future = pool.submit(_wait_for_port, "127.0.0.1", FRONTEND_PORT) if frontend else None
        api_ready = api_future.result()
        fe_ready = fe_future.result() if fe_future else False

    console.print()
    if api_ready:
        console.print(f"  [bold green]✓[/] API server  → http://localhost:{API_PORT}")
    else:
        console.print("  [bold red]✗[/] API server  failed to start")
    if frontend:
        if fe_ready:
            console.print(f"  [bold green]✓[/] Web UI      → http://localhost:{FRONTEND_PORT}")
        else:
            console.print("  [bold red]✗[/] Frontend    failed to start")

    console.print()
    console.print("[dim]Press Ctrl+C to stop both servers.[/]")
    console.print()

    try:
        # Wait for either process to exit (or user hits Ctrl+C)
        while True:
            if backend.poll() is not None:
                console.print("[yellow]Backend exited.[/]")
                break
            if frontend and frontend.poll() is not None:
                console.print("[yellow]Frontend exited.[/]")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down…[/]")
    finally:
        for proc in (backend, frontend):
            if proc:
                _kill_tree(proc)


def action_frontend() -> None:
    """Start only the Vite dev server."""
    fe = ROOT / "frontend"
    if not fe.exists():
        console.print("[red]frontend/ directory not found[/]")
        return
    console.print()
    _kill_port(FRONTEND_PORT)
    console.print(f"[bold]Starting frontend on http://localhost:{FRONTEND_PORT} …[/]\n")
    _run(["npm", "run", "dev"], cwd=fe)


def action_backend() -> None:
    """Start only the FastAPI server."""
    console.print()
    _kill_port(API_PORT)
    console.print(f"[bold]Starting API server on http://localhost:{API_PORT} …[/]\n")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "distortion_engine.api.app:app",
        "--reload",
        "--port",
        str(API_PORT),
    ]
    _run(cmd)


def action_demo() -> None:
    """Run the offline demo fixture."""
    console.print()
    console.print("[bold]Running offline demo…[/]\n")
    _run([sys.executable, "-m", "distortion_engine.cli", "demo"])


def action_test() -> None:
    """Run pytest."""
    console.print()
    console.print("[bold]Running tests…[/]\n")
    _run([sys.executable, "-m", "pytest", "-q"])


def action_lint() -> None:
    """Run ruff check + format check."""
    console.print()
    console.print("[bold]Running ruff…[/]\n")
    rc = _run([sys.executable, "-m", "ruff", "check", "."])
    _run([sys.executable, "-m", "ruff", "format", "--check", "."])
    if rc != 0:
        console.print("[yellow]Ruff found issues (see above)[/]")


def action_env_view() -> None:
    """Show which env vars are configured (never show values)."""
    status = _env_status()
    table = Table(title="Environment Variables", show_header=True, header_style="bold cyan")
    table.add_column("Variable")
    table.add_column("Status")
    for k, v in status.items():
        table.add_row(k, v)
    console.print()
    console.print(table)
    console.print(f"\n[dim]Config file: {ENV_FILE}[/]\n")


def action_env_edit() -> None:
    """Open .env in $EDITOR or print instructions."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        _run([editor, str(ENV_FILE)])
    else:
        console.print(f"\n[bold]Edit your .env file:[/]\n  [dim]{ENV_FILE}[/]\n")
        if not ENV_FILE.exists() and EXAMPLE_ENV.exists():
            console.print("[dim]Tip: copy .env.example to .env first[/]")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

MENU: list[tuple[str, str, str, callable]] = [
    ("1", "Launch Web UI", "Start backend + frontend", action_launch_web),
    ("2", "Frontend only", "Vite dev server on port 5173", action_frontend),
    ("3", "Backend only", "API server on port 8000", action_backend),
    ("4", "Run offline demo", "Execute the fixture demo (no provider needed)", action_demo),
    ("5", "Run tests", "pytest -q", action_test),
    ("6", "Lint", "ruff check + format check", action_lint),
    ("7", "Env vars", "View or edit .env configuration", action_env_view),
    ("q", "Quit", "", None),
]


def build_header() -> Panel:
    version = _version()
    title = Text(f"Distortion Engine  v{version}", style="bold white")
    subtitle = Text(
        "causal evaluation environment for hierarchical agent organizations",
        style="dim",
    )
    return Panel(Text.assemble(title, "\n", subtitle), border_style="blue", padding=(0, 2))


def build_menu_table() -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="bold cyan", width=3)
    table.add_column("name", style="bold", min_width=18)
    table.add_column("desc", style="dim")
    for key, name, desc, _ in MENU:
        table.add_row(key, name, desc)
    return table


def main() -> None:
    console.clear()
    console.print()
    console.print(build_header())
    console.print()

    # Show env status inline
    status = _env_status()
    model_status = status.get("DISTORTION_MODEL", "[dim]not set[/]")
    console.print(f"  Model: {model_status}")
    console.print()
    console.print(build_menu_table())
    console.print()

    while True:
        try:
            choice = console.input("[bold cyan]▸ [/]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        if choice in ("q", "quit", "exit"):
            console.print("[dim]Bye.[/]")
            break

        # "7" opens a sub-menu for env management
        if choice == "7":
            _env_sub_menu()
            console.print()
            console.print(build_menu_table())
            console.print()
            continue

        action = None
        for key, _, _, fn in MENU:
            if choice == key:
                action = fn
                break

        if action is None:
            console.print("[dim]Invalid choice — try again.[/]")
            continue

        action()
        console.print()
        console.print(build_menu_table())
        console.print()


def _env_sub_menu() -> None:
    """Sub-menu for env var management."""
    while True:
        console.print()
        console.print("[bold]Environment Variables[/]\n")
        console.print("  [cyan]1[/]  View status")
        console.print("  [cyan]2[/]  Edit .env file")
        console.print("  [cyan]b[/]  Back\n")

        try:
            choice = console.input("[bold cyan]▸ [/]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "b":
            break
        elif choice == "1":
            action_env_view()
        elif choice == "2":
            action_env_edit()
        else:
            console.print("[dim]Invalid choice.[/]")


if __name__ == "__main__":
    main()
