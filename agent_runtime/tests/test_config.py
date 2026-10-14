from pathlib import Path
import os
import sys
import tempfile

import pytest

from agent_runtime import agent_config as config
from agent_runtime import prepare_data


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    package = repository / "agent_runtime"
    package.mkdir(parents=True)
    root = repository / "ml"
    root.mkdir()
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "MARGARET_AI_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "APP", package)
    monkeypatch.setattr(config, "REPO", repository)
    monkeypatch.setattr(config, "PROJECT", root)
    return root, repository, package


def _settings_file(folder, key, model="gpt-5-mini"):
    location = folder / ".streamlit/secrets.toml"
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(f'OPENAI_API_KEY = "{key}"\nOPENAI_MODEL = "{model}"\n', encoding="utf-8")


def _manifest(root):
    location = root / "outputs/agent_data/manifest.json"
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text("{}", encoding="utf-8")


def test_artifact_root_environment_then_manifest_order(isolated, monkeypatch):
    root, repository, package = isolated
    assert config.artifact_root() == root
    _manifest(repository.parent)
    assert config.artifact_root() == repository.parent
    _manifest(repository)
    assert config.artifact_root() == repository
    _manifest(root)
    assert config.artifact_root() == root
    configured = repository / "configured"
    monkeypatch.setenv("MARGARET_AI_ROOT", str(configured))
    assert config.artifact_root() == configured.resolve()


def test_default_secret_order_and_environment_precedence(isolated, monkeypatch):
    root, repository, package = isolated
    _settings_file(root, "test-root-value")
    _settings_file(repository, "test-repo-value")
    _settings_file(package, "test-package-value")
    assert config.load_settings().api_key == "test-package-value"
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-value")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    settings = config.load_settings()
    assert settings.api_key == "test-environment-value" and settings.model == "test-model"
    assert "test-environment-value" not in repr(settings)


def test_explicit_settings_directory_never_falls_back(isolated):
    root, repository, package = isolated
    _settings_file(package, "must-not-be-used")
    settings = config.load_settings(project=root)
    assert settings.api_key == "" and settings.model == "gpt-5-mini"
    _settings_file(root, "explicit-test-value")
    assert config.load_settings(project=root).api_key == "explicit-test-value"


def test_invalid_secret_file_does_not_echo_contents(isolated, monkeypatch):
    root, _, _ = isolated
    location = root / ".streamlit/secrets.toml"
    location.parent.mkdir()
    location.write_text('OPENAI_API_KEY = "unterminated-test-value', encoding="utf-8")
    result = config.load_settings(project=root)
    assert result.api_key == "" and result.note
    assert "unterminated-test-value" not in repr(result)
    monkeypatch.setenv("OPENAI_API_KEY", "valid-environment-test-value")
    assert config.load_settings(project=root).api_key == "valid-environment-test-value"


@pytest.mark.parametrize("arguments,code", [(["--help"], 0), ([], 2), (["--project-root", "unused", "--chunksize", "999"], 2)])
def test_prepare_cli_help_and_invalid_arguments_do_not_prepare(arguments, code, monkeypatch):
    from agent_runtime import agent_data
    monkeypatch.setattr(agent_data, "prepare_data_cache", lambda *args, **kwargs: pytest.fail("Must not prepare data"))
    with pytest.raises(SystemExit) as raised:
        prepare_data.main(arguments)
    assert raised.value.code == code


def test_prepare_cli_passes_explicit_inputs_without_training(isolated, monkeypatch):
    from agent_runtime import agent_data
    root, _, _ = isolated
    source = root / "fixture.csv"
    source.write_text("fixture", encoding="utf-8")
    captured = []
    def prepare(*args, **kwargs):
        captured.append((args, kwargs))
        return {"complete": True, "dates": {"2025-04-29": {}}}
    monkeypatch.setattr(agent_data, "prepare_data_cache", prepare)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
    for name in ("TMP", "TEMP", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "PYTHONIOENCODING", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    assert prepare_data.main(["--project-root", str(root), "--source", str(source), "--chunksize", "1000", "--rebuild"]) == 0
    assert captured == [((root.resolve(),), {"chunksize": 1000, "rebuild": True, "source_path": source})]
    assert Path(os.environ["TMP"]).is_relative_to(root)
