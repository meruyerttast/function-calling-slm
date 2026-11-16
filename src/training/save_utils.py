"""Push trained adapters to HF Hub with verification.

V2-V5 adapters were lost because Colab `/content` is ephemeral and the manual
Drive copy in the notebooks had no fallback or verification. This module makes
the push the canonical save path: training fails fast at startup if the env is
not configured, and the push is verified before the script returns.
"""

import os
from pathlib import Path

REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def resolve_repo_id(cli_value: str | None) -> str | None:
    return cli_value or os.environ.get("HF_REPO_ID")


def assert_save_environment(hf_repo_id: str | None, allow_no_push: bool = False) -> str | None:
    """Validate HF env at training startup. Returns repo_id, or None if push disabled.

    Hard-fails by default. Pass allow_no_push=True only for local smoke tests.
    """
    if allow_no_push:
        print("WARNING: --no-push enabled. Adapter will save locally only.")
        return None

    repo_id = resolve_repo_id(hf_repo_id)
    if not repo_id:
        raise RuntimeError(
            "HF_REPO_ID not set. Refusing to start training because the adapter "
            "would be lost on Colab restart. Pass --hf-repo-id <user>/<repo> or "
            "set HF_REPO_ID env var. To bypass for local testing, pass --no-push."
        )
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN env var required to push adapter to HF Hub. "
            "In Colab: Tools -> User secrets -> add HF_TOKEN."
        )
    return repo_id


def push_adapter_to_hub(local_dir: str | Path, repo_id: str, run_name: str) -> str:
    """Upload adapter files to HF Hub and verify the upload landed."""
    from huggingface_hub import HfApi, create_repo

    local_dir = Path(local_dir)
    missing_local = [f for f in REQUIRED_FILES if not (local_dir / f).exists()]
    if missing_local:
        raise RuntimeError(f"Cannot push: missing {missing_local} in {local_dir}")

    token = os.environ["HF_TOKEN"]
    create_repo(repo_id, token=token, repo_type="model", private=True, exist_ok=True)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload {run_name}",
    )

    listed = set(api.list_repo_files(repo_id))
    missing_remote = [f for f in REQUIRED_FILES if f not in listed]
    if missing_remote:
        raise RuntimeError(
            f"Push verification failed: {missing_remote} not on hub for {repo_id}"
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Adapter pushed and verified: {url}")
    return url
