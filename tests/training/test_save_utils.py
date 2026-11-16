import pytest

from src.training.save_utils import assert_save_environment, resolve_repo_id


def test_assert_fails_without_repo_id(monkeypatch):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setenv("HF_TOKEN", "fake")
    with pytest.raises(RuntimeError, match="HF_REPO_ID not set"):
        assert_save_environment(None)


def test_assert_fails_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_REPO_ID", "user/repo")
    with pytest.raises(RuntimeError, match="HF_TOKEN env var required"):
        assert_save_environment(None)


def test_assert_returns_repo_id_from_cli(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    assert assert_save_environment("user/repo-cli") == "user/repo-cli"


def test_assert_returns_repo_id_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("HF_REPO_ID", "user/repo-env")
    assert assert_save_environment(None) == "user/repo-env"


def test_assert_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("HF_REPO_ID", "user/from-env")
    assert assert_save_environment("user/from-cli") == "user/from-cli"


def test_no_push_skips_validation(monkeypatch, capsys):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert assert_save_environment(None, allow_no_push=True) is None
    assert "WARNING" in capsys.readouterr().out


def test_resolve_repo_id_prefers_cli(monkeypatch):
    monkeypatch.setenv("HF_REPO_ID", "from-env")
    assert resolve_repo_id("from-cli") == "from-cli"
    assert resolve_repo_id(None) == "from-env"
