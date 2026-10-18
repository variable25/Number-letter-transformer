import ast
import os
import re

from model_training import Checkpoint_path

Root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repo root (tests/ -> ..)


# ---------- Helpers ----------
def read_text(name):
    with open(os.path.join(Root, name), encoding="utf-8") as file:
        return file.read()


def requirement_names():
    # Example: "torch==2.14.1+cpu" -> "torch"; skips comments, blanks and "--extra-index-url ..."
    names = []
    for line in read_text("requirements.txt").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.append(re.split(r"[=<>\[ ]", line)[0].lower())
    return names


def local_imports(start="app.py"):
    # Example: app.py -> translate.py -> attention.py -> data.py ... (only files in the repo root)
    to_visit, found = [start], set()
    while to_visit:
        name = to_visit.pop()
        if name in found:
            continue
        found.add(name)
        for node in ast.walk(ast.parse(read_text(name))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                file = module.split(".")[0] + ".py"
                if os.path.exists(os.path.join(Root, file)):
                    to_visit.append(file)
    return found


def dockerignore_lines():
    lines = [line.strip() for line in read_text(".dockerignore").splitlines()]
    return {line for line in lines if line and not line.startswith("#")}


# ---------- requirements.txt ----------
def test_requirements_has_serving_packages():
    names = requirement_names()
    for package in ["torch", "fastapi", "uvicorn"]:
        assert package in names


def test_requirements_torch_is_cpu_build():
    # Example: torch==2.14.1+cpu from PyTorch's CPU index, not the +cu130 build in .venv
    text = read_text("requirements.txt")
    assert re.search(r"^torch==\S+\+cpu$", text, flags=re.MULTILINE)
    assert "download.pytorch.org/whl/cpu" in text


def test_requirements_has_no_test_or_plot_packages():
    names = requirement_names()
    for package in ["pytest", "httpx", "httpx2", "matplotlib"]:
        assert package not in names


def test_requirements_pins_every_package():
    # "==" on every line -> the same image every time we build
    for line in read_text("requirements.txt").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            assert "==" in line, line


# ---------- Dockerfile ----------
def test_dockerfile_uses_slim_python_312():
    assert re.search(r"^FROM python:3\.12-slim", read_text("Dockerfile"), flags=re.MULTILINE)


def test_dockerfile_copies_exactly_what_app_needs():
    # the COPY line that has app.py, minus "COPY" and the "./" at the end
    copy_line = next(line for line in read_text("Dockerfile").splitlines()
                     if line.startswith("COPY") and "app.py" in line)
    copied = set(copy_line.split()[1:-1])
    assert copied == local_imports()
    assert "attention_viz.py" not in copied


def test_dockerfile_copies_best_pt_and_sets_model_path():
    # Example: Checkpoint_path "checkpoints\best.pt" on Windows -> "/app/checkpoints/best.pt" in Linux
    text = read_text("Dockerfile")
    linux_path = Checkpoint_path.replace(os.sep, "/")
    assert f"COPY {linux_path} {linux_path}" in text
    assert re.search(r"MODEL_PATH=(\S+)", text).group(1) == "/app/" + linux_path


def test_dockerfile_installs_before_copying_code():
    # layer caching: shopping list -> pip install -> code. Editing app.py must not re-download torch
    text = read_text("Dockerfile")
    assert text.index("COPY requirements.txt") < text.index("pip install") < text.index("COPY app.py")


def test_dockerfile_listens_on_all_interfaces_and_port_env():
    text = read_text("Dockerfile")
    cmd_line = next(line for line in text.splitlines() if line.startswith("CMD"))
    assert "--host 0.0.0.0" in cmd_line
    assert "${PORT}" in cmd_line
    assert "PORT=8080" in text


def test_dockerfile_healthcheck_hits_health():
    text = read_text("Dockerfile")
    assert "HEALTHCHECK" in text
    assert "/health" in text[text.index("HEALTHCHECK"):]


# ---------- .dockerignore ----------
def test_dockerignore_keeps_junk_out():
    lines = dockerignore_lines()
    for entry in [".venv/", ".git/", "__pycache__/", ".pytest_cache/", "attention_maps/", "tests/"]:
        assert entry in lines


def test_dockerignore_lets_best_pt_in():
    # skip the folder, then "!" lets best.pt back in; plain "checkpoints/" would block it
    lines = dockerignore_lines()
    assert "checkpoints/" not in lines
    assert "checkpoints/*" in lines
    assert "!checkpoints/best.pt" in lines