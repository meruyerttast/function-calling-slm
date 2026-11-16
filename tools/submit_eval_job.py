"""Submit a BFCL eval HF Job for a given variant.

Reuses the validated 2026-04-28 recipe (see memory: hf-jobs-bfcl-recipe).
Pre-flight checks every precondition before submitting — adapter exists on
Hub, all required secrets present, expected token namespace matches.

Usage from Colab:
    !python tools/submit_eval_job.py \\
        --variant v6 \\
        --adapter-repo arnursb/info290-v6-grpo-curriculum

Usage with custom GH PAT (e.g. from Colab userdata):
    GH_PAT=ghp_xxx python tools/submit_eval_job.py --variant v6 ...

Skip agentic bucket (cheaper, ~10 min vs ~45 min):
    python tools/submit_eval_job.py --variant v6 --no-agentic ...
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, get_token

GIT_REPO = "arnursabet/info290-function-calling-slm"
COMMITTER_NAME = "Arnur Sabet"
COMMITTER_EMAIL = "sabet@berkeley.edu"
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "a10g-large"
TIMEOUT = "3h"


def build_bash_script(variant: str, adapter_repo: str, with_agentic: bool) -> str:
    buckets = (
        "irrelevance live multi_turn agentic" if with_agentic
        else "irrelevance live multi_turn"
    )
    return f"""
set -euo pipefail
apt-get update -qq && apt-get install -y -qq git ca-certificates libsndfile1
git clone --depth 1 https://x-access-token:${{GH_TOKEN}}@github.com/{GIT_REPO}.git /workspace/repo
cd /workspace/repo

pip install --no-cache-dir --quiet pyyaml
pip install --no-cache-dir --quiet "vllm==0.15.1"
pip install --no-cache-dir --quiet --no-deps "bfcl-eval @ git+https://github.com/arnursabet/gorilla.git@f64471b7bf7b4879c1434804f6c3bb5ff5c9bc48#subdirectory=berkeley-function-call-leaderboard"
pip install --no-cache-dir --quiet "numpy<2" requests tqdm pandas "pydantic>=2.8.2" "python-dotenv>=1.0.1" \\
  "tree_sitter==0.21.3" "tree-sitter-java==0.21.0" "tree-sitter-javascript==0.21.4" \\
  "typer>=0.12.5" "tabulate>=0.9.0" overrides "tenacity>=8.5.0" \\
  "datamodel-code-generator==0.25.7" "filelock==3.20.0" "networkx==3.3" \\
  "openai>=1.86.0" "anthropic>=0.75.0" "mistralai==1.7.0" "cohere==5.18.0" \\
  "google-genai>=1.52.0" qwen-agent boto3 "faiss-cpu==1.11.0" google-search-results \\
  html2text "rank_bm25==0.2.2" "sentence-transformers>=2.7.0" "writer-sdk>=2.1.0" \\
  "mpmath==1.3.0" beautifulsoup4 soundfile

python3 -c "from huggingface_hub import snapshot_download; p = snapshot_download(repo_id='{adapter_repo}', token='${{HF_TOKEN}}'); print('ADAPTER_DIR=' + p)" > /tmp/adapter.log
ADAPTER_DIR=$(grep '^ADAPTER_DIR=' /tmp/adapter.log | cut -d= -f2)
echo "Adapter at $ADAPTER_DIR"

export BFCL_VLLM_MAX_MODEL_LEN=4096
export BFCL_VLLM_MAX_NUM_SEQS=64
export BFCL_HANDLER_DTYPE=bfloat16

for bucket in {buckets}; do
    python3 -m src.evaluation.bfcl --variant {variant} --test-category "$bucket" --adapter "$ADAPTER_DIR" --adapter-source "{adapter_repo}"
done

git config user.name "{COMMITTER_NAME}"
git config user.email "{COMMITTER_EMAIL}"
BRANCH="eval/{variant}-buckets-$(date -u +%Y%m%d-%H%M)"
git checkout -b "$BRANCH"
git add results/{variant}/
git commit -m "{variant}: add eval buckets ({buckets})"
git push https://x-access-token:${{GH_TOKEN}}@github.com/{GIT_REPO}.git "$BRANCH"
echo "DONE. Branch: $BRANCH"
"""


def preflight_check(adapter_repo: str, gh_pat: str | None) -> tuple[str, str]:
    """Verify all preconditions. Exits with code 1 on failure.

    Returns (hf_token, gh_pat).
    """
    print("=== Pre-flight checks ===")

    hf_token = get_token()
    if not hf_token:
        sys.exit("ERROR: no HF token. Run `hf auth login` or set HF_TOKEN env var.")
    api = HfApi(token=hf_token)
    try:
        whoami = api.whoami()
        print(f"  HF token OK: {whoami['name']}")
    except Exception as e:
        sys.exit(f"ERROR: HF token invalid: {e}")

    try:
        info = api.model_info(adapter_repo, token=hf_token)
        files = [f.rfilename for f in info.siblings]
        if "adapter_config.json" not in files:
            sys.exit(f"ERROR: {adapter_repo} missing adapter_config.json. Files: {files}")
        if not any(f.endswith(".safetensors") for f in files):
            sys.exit(f"ERROR: {adapter_repo} missing .safetensors. Files: {files}")
        print(f"  Adapter OK: {adapter_repo} ({len(files)} files, including adapter_config.json + safetensors)")
    except Exception as e:
        sys.exit(f"ERROR: cannot access {adapter_repo}: {e}")

    if not gh_pat:
        sys.exit(
            "ERROR: GH_PAT required. Pass via --gh-pat, GH_PAT env var, or in Colab:\n"
            "    from google.colab import userdata\n"
            "    import os\n"
            "    os.environ['GH_PAT'] = userdata.get('GITHUB_TOKEN')"
        )
    if not gh_pat.startswith(("ghp_", "github_pat_")):
        print(f"  WARNING: GH_PAT doesn't start with ghp_/github_pat_ (got '{gh_pat[:8]}...'). Proceeding anyway.")
    else:
        print(f"  GH PAT OK: {gh_pat[:8]}... (length {len(gh_pat)})")

    return hf_token, gh_pat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="Variant name (v4, v5, v6)")
    parser.add_argument("--adapter-repo", required=True, help="HF Hub adapter repo (e.g. arnursb/info290-v6-grpo-curriculum)")
    parser.add_argument("--no-agentic", action="store_true", help="Skip agentic bucket (cheaper, ~10 min, no Serper needed)")
    parser.add_argument("--gh-pat", default=None, help="GitHub PAT for pushing eval results branch (else from GH_PAT env)")
    parser.add_argument("--dry-run", action="store_true", help="Print bash script without submitting")
    args = parser.parse_args()

    gh_pat = args.gh_pat or os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    serper_key = os.environ.get("SERPER_API_KEY")

    hf_token, gh_pat = preflight_check(args.adapter_repo, gh_pat)

    with_agentic = not args.no_agentic
    if with_agentic and not serper_key:
        sys.exit(
            "ERROR: SERPER_API_KEY env var required for agentic bucket "
            "(web_search needs Serper). Pass --no-agentic to skip, or set:\n"
            "    os.environ['SERPER_API_KEY'] = userdata.get('SERPER_API_KEY')"
        )
    bash = build_bash_script(args.variant, args.adapter_repo, with_agentic)

    print()
    print("=== Submission summary ===")
    print(f"  Variant:       {args.variant}")
    print(f"  Adapter:       {args.adapter_repo}")
    print(f"  Buckets:       {'irrelevance live multi_turn agentic' if with_agentic else 'irrelevance live multi_turn'}")
    print(f"  Agentic:       {'YES (Serper enabled)' if with_agentic else 'NO'}")
    print(f"  Image:         {DOCKER_IMAGE}")
    print(f"  Flavor:        {FLAVOR}")
    print(f"  Timeout:       {TIMEOUT}")
    print(f"  Push to:       branch eval/{args.variant}-buckets-<timestamp> on {GIT_REPO}")
    print()

    if args.dry_run:
        print("--- BASH SCRIPT (dry run) ---")
        print(bash)
        return

    api = HfApi(token=hf_token)
    secrets = {"HF_TOKEN": hf_token, "GH_TOKEN": gh_pat}
    if with_agentic:
        secrets["SERPER_API_KEY"] = serper_key

    print("Submitting...")
    job = api.run_job(
        image=DOCKER_IMAGE,
        command=["bash", "-c", bash],
        flavor=FLAVOR,
        timeout=TIMEOUT,
        secrets=secrets,
    )
    print()
    print("=== Submitted ===")
    print(f"  Job ID:  {job.id}")
    print(f"  URL:     https://huggingface.co/jobs/{job.namespace}/{job.id}")
    print(f"  ETA:     ~30-45 min with agentic, ~10 min without")


if __name__ == "__main__":
    main()
