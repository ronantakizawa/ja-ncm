"""Upload a trained ja-NCM model to a PRIVATE Hugging Face repo, with a model card.

  python scripts/hf_upload.py --kind ncm --ckpt runs/ja_en_ncm/last.pt --repo <hf-user>/<repo-name> \
      --results results/ncm_final --title "ja-NCM ja->en (WMT14 NCM fine-tune)"

Weights are written as safetensors (no pickle). Repos are created with private=True, and an
existing public repo is refused rather than overwritten. The token is read from
$HF_TOKEN_PATH (mode-600 file on the training node) or the local Hugging Face login.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jancm import ROOT  # noqa: E402
from jancm.loading import load_state, resolve  # noqa: E402


def save_safetensors(state_dict, path):
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous().clone() for k, v in state_dict.items()}, path)


def results_table(rdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        d = json.load(open(f))
        if "bleu" in d:
            rows.append(f"| {d['test']} | {d['n']} | {d['bleu']:.2f} | {d['chrf']:.2f} | "
                        f"{d.get('comet22', float('nan')):.2f} | {d['ms_per_sentence']:.1f} | {d['nfe_mean']:.1f} |")
    if not rows:
        return "(no results)"
    return ("| Test | n | BLEU | chrF | COMET-22 | ms/sentence (H100, batch 16) | NFE |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))


CARD = """---
license: {license}
language: [ja, en]
tags: [next-concept-modeling, ncm, translation, research]
---
# {title}

Private research checkpoint from the ja-NCM project: Next-Concept Modeling
([Kumbong et al., 2026](https://arxiv.org/abs/2610.14162), code MIT) applied to Japanese.
{description}

## Results
{table}

BLEU is sacreBLEU (13a for English output, ja-mecab for Japanese) and chrF is sacreBLEU chrF;
COMET-22 is Unbabel/wmt22-comet-da (x100). Decoding is greedy.

## Files
{files}

## Training data and licences
{data}

Upstream NCM code and released checkpoints: MIT (HazyResearch/ncm). Because training data
includes share-alike corpora, these weights are marked {license}.

## Loading
Use the ja-NCM project code (vendored NCM model classes). `config.yaml` is the full training
config; weights load with `safetensors.torch.load_file` and `load_state_dict(strict=True)`.
"""

DATA_A = """- KFTT (CC BY-SA 3.0), Mitsua/wikidata-parallel-descriptions-en-ja descriptions and labels (CC0),
  Tatoeba via OPUS (CC BY 2.0 FR), ALT (CC BY 4.0). Test: FLORES+ devtest, NTREX-128 (CC BY-SA 4.0),
  KFTT test, ALT test."""


def scrub_config(cfg):
    """Drop machine-specific settings before publishing a config: every *.sync_cmd (checkpoint
    sync command, which names the training storage) is blanked."""
    from omegaconf import DictConfig, ListConfig

    def walk(node):
        if isinstance(node, DictConfig):
            for k in list(node.keys()):
                if k == "sync_cmd":
                    node[k] = ""
                else:
                    walk(node[k])
        elif isinstance(node, ListConfig):
            for v in node:
                walk(v)
    walk(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ncm", "ar"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--license", default="cc-by-sa-4.0")
    ap.add_argument("--data", default="phase_a")
    ap.add_argument("--include-tokenizer", action="store_true", help="also upload the frozen concept tokenizer")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    tok_path = os.environ.get("HF_TOKEN_PATH")
    token = open(tok_path).read().strip() if tok_path and os.path.exists(tok_path) else None
    api = HfApi(token=token)
    try:
        info = api.model_info(a.repo)
        if not info.private:
            raise SystemExit(f"{a.repo} exists and is PUBLIC; refusing to upload")
    except SystemExit:
        raise
    except Exception:
        pass
    api.create_repo(a.repo, private=True, exist_ok=True, repo_type="model")

    st = load_state(a.ckpt)
    cfg = OmegaConf.create(st["config"])
    with tempfile.TemporaryDirectory() as d:
        files = []
        name = "generator.safetensors" if a.kind == "ncm" else "model.safetensors"
        save_safetensors(st["model"], os.path.join(d, name))
        files.append(f"- `{name}`: {a.kind.upper()} weights (step {st.get('step')})")
        OmegaConf.save(scrub_config(cfg), os.path.join(d, "config.yaml"))
        files.append("- `config.yaml`: full training config")
        if a.include_tokenizer:
            tsd = load_state(resolve(cfg.checkpoints.tokenizer, ROOT))["model"]
            save_safetensors(tsd, os.path.join(d, "tokenizer.safetensors"))
            files.append("- `tokenizer.safetensors`: frozen multiscale VQ-VAE concept tokenizer")
        if a.results and os.path.isdir(a.results):
            shutil.copytree(a.results, os.path.join(d, "results"))
            files.append("- `results/`: per-test-set scores and outputs")
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(CARD.format(license=a.license, title=a.title, description=a.description,
                                table=results_table(a.results) if a.results else "(not evaluated)",
                                files="\n".join(files), data=DATA_A if a.data == "phase_a" else a.data))
        api.upload_folder(folder_path=d, repo_id=a.repo, repo_type="model",
                          commit_message=f"upload {os.path.basename(a.ckpt)} (step {st.get('step')})")
    assert api.model_info(a.repo).private, "repo is not private after upload"
    print(f"uploaded privately: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
