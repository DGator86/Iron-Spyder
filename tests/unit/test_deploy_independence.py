"""Iron-Spyder's VPS deploy has no runtime dependency on legacy trees.

Scans shipped units, env/config templates, and the VPS package, and fails on
any 0DTE/legacy path or unit name. Also asserts the VPS service set is
complete and that every ExecStart names a real CLI command.
"""

from __future__ import annotations

import ast
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

REQUIRED_UNITS = (
    "iron-spyder-supervisor.service",
    "iron-spyder-dashboard-api.service",
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

_SPYDER_VPS_BIN = "/opt/iron-spyder/venv/bin/spyder-vps"


def _deploy_files() -> list[Path]:
    return sorted(p for p in _DEPLOY.iterdir() if p.is_file())


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


def test_update_timer_has_matching_service() -> None:
    assert (_DEPLOY / "iron-spyder-update.service").is_file()
    assert (_DEPLOY / "iron-spyder-update.timer").is_file()


@pytest.mark.parametrize(
    "unit",
    [u for u in REQUIRED_UNITS if u.endswith(".service") and u != "iron-spyder-update.service"],
)
def test_runtime_services_run_as_iron_spyder_user(unit: str) -> None:
    text = (_DEPLOY / unit).read_text(encoding="utf-8")
    assert "User=iron-spyder" in text, unit
    assert "Group=iron-spyder" in text, unit
    assert "NoNewPrivileges=true" in text, unit
    assert "StateDirectory=iron-spyder" in text, unit
    assert "IRON_SPYDER_STATE_ROOT=%S/iron-spyder" in text, unit


def test_dashboard_api_binds_loopback_only() -> None:
    text = (_DEPLOY / "iron-spyder-dashboard-api.service").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
    assert "0.0.0.0" not in text
    assert "ReadOnlyPaths=/var/lib/iron-spyder" in text


def test_dashboard_api_cannot_mutate_state_root() -> None:
    text = (_DEPLOY / "iron-spyder-dashboard-api.service").read_text(encoding="utf-8")
    writable = {
        path
        for line in text.splitlines()
        if line.strip().startswith("ReadWritePaths=")
        for path in line.strip().split("=", 1)[1].split()
    }
    assert "/var/lib/iron-spyder" not in writable


def test_no_unit_enables_live_trading() -> None:
    for path in _deploy_files():
        text = path.read_text(encoding="utf-8")
        assert "IRON_SPYDER_ALLOW_LIVE=1" not in text, path.name
        assert "--mode live" not in text, path.name


def _cli_commands() -> set[str]:
    source = (_VPS / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_VPS / "cli.py"))
    commands: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, ast.In) for op in node.ops):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "cmd"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Set):
                for element in comparator.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        commands.add(element.value)
    return commands


def _exec_start_tokens(text: str) -> list[str]:
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("ExecStart="):
            return stripped.removeprefix("ExecStart=").split()
    return []


def _exec_start_subcommand(text: str) -> str | None:
    tokens = _exec_start_tokens(text)
    if not tokens or tokens[0] != _SPYDER_VPS_BIN:
        return None
    for token in tokens[1:]:
        if not token.startswith("-"):
            return token
    return None


def _units_invoking_the_cli() -> list[str]:
    return sorted(
        p.name
        for p in _DEPLOY.glob("*.service")
        if _exec_start_tokens(p.read_text(encoding="utf-8"))[:1] == [_SPYDER_VPS_BIN]
    )


@pytest.mark.parametrize("unit", _units_invoking_the_cli())
def test_every_shipped_unit_invokes_a_real_cli_command(unit: str) -> None:
    path = _DEPLOY / unit
    subcommand = _exec_start_subcommand(path.read_text(encoding="utf-8"))
    assert subcommand, f"{unit} has no parseable ExecStart subcommand"
    assert subcommand in _cli_commands(), (
        f"{unit} runs `spyder-vps {subcommand}`, which spy_der.vps.cli.main "
        f"does not dispatch"
    )


def test_every_runtime_unit_is_audited_by_the_cli_check() -> None:
    audited = set(_units_invoking_the_cli())
    everything = {p.name for p in _DEPLOY.glob("*.service")}
    assert everything - audited == {"iron-spyder-update.service"}


def test_units_that_run_a_script_ship_that_script() -> None:
    text = (_DEPLOY / "iron-spyder-update.service").read_text(encoding="utf-8")
    tokens = _exec_start_tokens(text)
    scripts = [t for t in tokens if t.endswith(".sh")]
    assert scripts
    for script in scripts:
        name = Path(script).name
        assert (_DEPLOY / name).is_file(), f"update unit runs {script}, not shipped"


def test_deploy_scripts_are_shipped_and_executable() -> None:
    for name in ("remote-deploy.sh", "self-update.sh"):
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


def test_remote_deploy_installs_the_package_into_the_venv() -> None:
    text = _remote_deploy()
    assert "pip" in text and "install" in text
    assert "venv/bin/pip" in text


def test_remote_deploy_creates_every_declared_state_directory() -> None:
    text = _remote_deploy()
    for name in REQUIRED_STATE_DIRS:
        assert name in text, f"state directory {name} is never created"


def test_remote_deploy_owns_state_as_the_service_user() -> None:
    text = _remote_deploy()
    assert 'SVC_USER=iron-spyder' in text
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
        if "ENV_FILE" in stripped and ("install " in stripped or "cp " in stripped):
            raise AssertionError(f"deploy writes the secrets file: {stripped}")


def test_self_update_runs_the_fetched_deploy_script_not_the_stale_one() -> None:
    text = (_DEPLOY / "self-update.sh").read_text(encoding="utf-8")
    assert "git -C \"$APP_DIR\" show" in text
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


def test_config_template_declares_every_state_directory() -> None:
    text = (_DEPLOY / "config.yaml.example").read_text(encoding="utf-8")
    for name in REQUIRED_STATE_DIRS:
        assert f"- {name}" in text, f"state directory {name} not declared"


def test_pyproject_exposes_spyder_vps_console_script() -> None:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'spyder-vps = "spy_der.vps.cli:main"' in text
