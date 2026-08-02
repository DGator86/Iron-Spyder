"""Iron-Spyder deploy is a dedicated CPU VPS — no legacy trees, no GPU.

Scans shipped units, compose files, env/config templates, and the VPS package.
Fails on 0DTE/SPY-DER path leakage, GPU/CUDA wiring, or incomplete service set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY = _ROOT / "deploy"
_VPS = _ROOT / "spy_der" / "vps"

FORBIDDEN_PATHS = (
    "/opt/zerodte",
    "/var/lib/zerodte",
    "/etc/zerodte",
    "/opt/spy-der",
    "/var/lib/spy-der",
    "/etc/spy-der",
)

GPU_MARKERS = (
    "nvidia/",
    "nvidia.com",
    "runtime: nvidia",
    "deploy.resources.reservations.devices",
    "capabilities: [gpu]",
    "cuda:",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "FROM nvidia",
)

REQUIRED_UNITS = (
    "iron-spyder.service",
    "iron-spyder-update.service",
    "iron-spyder-update.timer",
)

REQUIRED_STATE_DIRS = (
    "health",
    "decisions",
    "positions",
    "audit",
    "reports",
    "configs",
)


def _deploy_files() -> list[Path]:
    return sorted(p for p in _DEPLOY.iterdir() if p.is_file())


def _compose_files() -> list[Path]:
    return [
        _ROOT / "docker-compose.yml",
        _ROOT / "docker-compose.research.yml",
        _ROOT / "Dockerfile",
    ]


def test_no_deploy_file_references_a_legacy_path() -> None:
    offenders: list[str] = []
    for path in _deploy_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text:
                offenders.append(f"{path.name}: {forbidden}")
    assert not offenders, offenders


def test_no_deploy_file_declares_a_zerodte_unit_or_user() -> None:
    offenders: list[str] = []
    for path in _deploy_files():
        if path.name.startswith("zerodte"):
            offenders.append(f"unit file named {path.name}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "zerodte" in stripped.lower():
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, offenders


def test_vps_package_references_no_legacy_filesystem_path() -> None:
    offenders: list[str] = []
    for path in sorted(_VPS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text:
                offenders.append(f"{path.relative_to(_ROOT)}: {forbidden}")
    assert not offenders, offenders


@pytest.mark.parametrize("unit", REQUIRED_UNITS)
def test_required_unit_is_shipped(unit: str) -> None:
    assert (_DEPLOY / unit).is_file(), f"missing {unit}"


def test_live_unit_is_docker_compose() -> None:
    text = (_DEPLOY / "iron-spyder.service").read_text(encoding="utf-8")
    assert "docker compose" in text
    assert "docker-compose.yml" in text
    assert "User=spy-der" not in text
    assert "/opt/spy-der" not in text


def test_update_timer_has_matching_service() -> None:
    assert (_DEPLOY / "iron-spyder-update.service").is_file()
    assert (_DEPLOY / "iron-spyder-update.timer").is_file()


def test_no_unit_enables_live_trading() -> None:
    for path in _deploy_files():
        text = path.read_text(encoding="utf-8")
        assert "IRON_SPYDER_ALLOW_LIVE=1" not in text, path.name
        assert "--mode live" not in text, path.name


def test_compose_and_dockerfile_are_cpu_only() -> None:
    offenders: list[str] = []
    for path in _compose_files():
        text = path.read_text(encoding="utf-8").lower()
        for marker in GPU_MARKERS:
            if marker.lower() in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, offenders


def test_compose_live_stack_binds_loopback_only() -> None:
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8000:8000" in text
    assert "127.0.0.1:8501:8501" in text
    assert "127.0.0.1:8788:8788" in text
    assert "0.0.0.0:8000" not in text
    assert "iron-spyder:cpu" in text
    assert "NVIDIA_VISIBLE_DEVICES: void" in text
    # Host state tree, not an anonymous named volume that hides deploy.json.
    assert "${IRON_SPYDER_STATE_ROOT:-/var/lib/iron-spyder}:/var/lib/iron-spyder" in text
    assert "iron-spyder-state:" not in text


def test_research_compose_is_separate_and_cpu_only() -> None:
    text = (_ROOT / "docker-compose.research.yml").read_text(encoding="utf-8")
    assert "iron-spyder-research" in text
    assert "IRON_SPYDER_MODE: research" in text
    assert "NVIDIA_VISIBLE_DEVICES: void" in text
    assert "ports:" not in text


def test_dockerfile_disables_cuda_visibility() -> None:
    text = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "NVIDIA_VISIBLE_DEVICES=void" in text
    assert "python:3.12" in text
    assert "FROM python:" in text
    assert "FROM nvidia" not in text.lower()


def test_cpu_vps_guidance_is_shipped() -> None:
    text = (_DEPLOY / "CPU_VPS.md").read_text(encoding="utf-8")
    assert "8" in text and "32 GB" in text
    assert "no GPU" in text.lower() or "Do not pay for a GPU" in text
    assert "Docker Compose" in text
    assert "second CPU" in text or "Research worker" in text


def test_deploy_scripts_are_shipped_and_executable() -> None:
    for name in ("remote-deploy.sh", "self-update.sh", "backup.sh"):
        path = _DEPLOY / name
        assert path.is_file(), name
        assert path.stat().st_mode & 0o111, f"{name} is not executable"


def _remote_deploy() -> str:
    return (_DEPLOY / "remote-deploy.sh").read_text(encoding="utf-8")


def test_remote_deploy_installs_every_required_unit() -> None:
    text = _remote_deploy()
    for unit in REQUIRED_UNITS:
        stem = unit.removesuffix(".service").removesuffix(".timer")
        assert stem in text, f"{unit} is never installed by remote-deploy.sh"


def test_remote_deploy_uses_docker_compose() -> None:
    text = _remote_deploy()
    assert "docker compose" in text
    assert "docker-compose.yml" in text
    assert "no NVIDIA/CUDA" in text or "CPU" in text


def test_remote_deploy_creates_every_declared_state_directory() -> None:
    text = _remote_deploy()
    for name in REQUIRED_STATE_DIRS:
        assert name in text, f"state directory {name} is never created"


def test_remote_deploy_owns_state_as_the_service_user() -> None:
    text = _remote_deploy()
    assert "SVC_USER=iron-spyder" in text
    assert 'chown -R "$SVC_USER:$SVC_USER" "$STATE_DIR"' in text


def test_remote_deploy_never_starts_units_without_the_secrets_file() -> None:
    text = _remote_deploy()
    assert "/etc/iron-spyder/iron-spyder.env" in text
    assert "not found" in text


def test_remote_deploy_does_not_write_the_secrets_file() -> None:
    for line in _remote_deploy().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Copying ENV_FILE -> compose .env is fine; writing ENV_FILE from example is not.
        writes_secrets = (
            "ENV_FILE" in stripped
            and ("install " in stripped or "cp " in stripped)
            and "iron-spyder.env.example" in stripped
            and stripped.rstrip().endswith("$ENV_FILE")
        )
        if writes_secrets:
            raise AssertionError(f"deploy writes the secrets file: {stripped}")


def test_self_update_runs_the_fetched_deploy_script_not_the_stale_one() -> None:
    text = (_DEPLOY / "self-update.sh").read_text(encoding="utf-8")
    assert 'git -C "$APP_DIR" show' in text
    assert "deploy/remote-deploy.sh" in text


def test_self_update_is_a_noop_when_already_current() -> None:
    text = (_DEPLOY / "self-update.sh").read_text(encoding="utf-8")
    assert 'if [ "$local_sha" = "$remote_sha" ]' in text


def test_update_timer_polls_on_a_bounded_interval() -> None:
    text = (_DEPLOY / "iron-spyder-update.timer").read_text(encoding="utf-8")
    assert "OnUnitActiveSec=" in text
    assert "OnBootSec=" in text


def test_remote_deploy_enables_the_self_update_timer() -> None:
    text = _remote_deploy()
    assert "enable --now iron-spyder-update.timer" in text


def test_deploy_targets_this_repo() -> None:
    text = _remote_deploy()
    assert "Iron-Spyder.git" in text


def test_env_template_exists_and_carries_no_secret_values() -> None:
    path = _DEPLOY / "iron-spyder.env.example"
    assert path.is_file()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if any(token in key for token in ("KEY", "PASSWORD", "SECRET", "TOKEN")):
            assert value == "", f"{key} must ship empty, got {value!r}"


def test_env_template_defaults_to_paper_with_live_disabled() -> None:
    text = (_DEPLOY / "iron-spyder.env.example").read_text(encoding="utf-8")
    assert "IRON_SPYDER_MODE=paper" in text
    assert "IRON_SPYDER_ALLOW_LIVE=0" in text
    assert "NVIDIA_VISIBLE_DEVICES=void" in text


def test_config_template_declares_every_state_directory() -> None:
    text = (_DEPLOY / "config.yaml.example").read_text(encoding="utf-8")
    for name in REQUIRED_STATE_DIRS:
        assert f"- {name}" in text, f"state directory {name} not declared"


def test_pyproject_exposes_spyder_vps_console_script() -> None:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'spyder-vps = "spy_der.vps.cli:main"' in text


def test_compose_env_example_is_shipped() -> None:
    text = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "IRON_SPYDER_MODE=paper" in text
    assert "IRON_SPYDER_ALLOW_LIVE=0" in text
