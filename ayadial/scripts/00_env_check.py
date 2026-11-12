"""Phase 0: check versions, device, HF login and access to every repo we need."""
import json
import platform
import sys

import torch
import transformers
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from common import DAY1, pick_device

REPOS = [
    ("CohereLabs/tiny-aya-global", "model", True),
    ("walledai/XSTest", "dataset", True),
    ("walledai/HarmBench", "dataset", True),
    ("JailbreakBench/JBB-Behaviors", "dataset", False),
    ("tatsu-lab/alpaca", "dataset", False),
    ("bench-llm/or-bench", "dataset", False),
    ("Qwen/Qwen3Guard-Gen-0.6B", "model", False),
]


def url(repo, kind):
    return f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"


def main():
    api = HfApi()
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": pick_device(),
    }
    try:
        report["hf_user"] = api.whoami()["name"]
    except Exception as err:
        print(f"Not logged in to Hugging Face ({err}). Run: hf auth login")
        sys.exit(1)

    access, missing = {}, []
    for repo, kind, gated in REPOS:
        try:
            api.auth_check(repo, repo_type=kind)
            access[repo] = "ok"
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as err:
            access[repo] = f"{type(err).__name__}"
            missing.append((repo, url(repo, kind), gated))
    report["access"] = access

    DAY1.mkdir(parents=True, exist_ok=True)
    with open(DAY1 / "env.json", "w") as f:
        json.dump(report, f, indent=2)

    for key, value in report.items():
        print(f"{key}: {value}")
    if missing:
        print("\nACCESS MISSING. Open and accept:")
        for repo, link, _ in missing:
            print(f"  {repo}: {link}")
        sys.exit(1)
    print("\nEnvironment OK.")


if __name__ == "__main__":
    main()
