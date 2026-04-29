from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable


NATURELM_DOMAIN_TOKENS = [
    "<molecule>",
    "</molecule>",
    "<material>",
    "</material>",
    "<protein>",
    "</protein>",
    "<dna>",
    "</dna>",
    "<rna>",
    "</rna>",
    "<antibody>",
    "</antibody>",
    "<sg>",
]

UNIGENX_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<coord>",
    "<orderxyz>",
    "<orderxzy>",
    "<orderyxz>",
    "<orderyzx>",
    "<orderzxy>",
    "<orderzyx>",
]


def clone_or_update(repo_url: str, target_dir: str | Path, ref: str | None = None) -> Path:
    target = Path(target_dir)
    if target.exists():
        if (target / ".git").exists():
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref or "main"], check=False)
        return target
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(target)])
    subprocess.run(cmd, check=True)
    return target


def read_git_commit(repo_dir: str | Path) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, capture_output=True, timeout=5, check=True)
        return proc.stdout.strip()
    except Exception:
        return ""


def naturelm_tokens_from_sfm(repo_dir: str | Path) -> list[str]:
    root = Path(repo_dir)
    tokens: set[str] = set(NATURELM_DOMAIN_TOKENS)
    for rel in ["README.md", "NatureLM/README.md"]:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens.update(re.findall(r"</?[A-Za-z][A-Za-z0-9_:-]*>", text))
        tokens.update(f"<sg{match}>" for match in re.findall(r"<sg(\d+)>", text))
    return sorted(tokens)


def unigenx_tokens_from_repo(repo_dir: str | Path) -> list[str]:
    root = Path(repo_dir)
    tokens: set[str] = set(UNIGENX_SPECIAL_TOKENS)
    data_dir = root / "unigenx" / "data"
    for path in sorted(data_dir.glob("dict*.txt")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = line.strip()
            if token:
                tokens.add(token)
                tokens.add(f"UNIGENX:TOK:{token}")
    tokenizer_path = data_dir / "tokenizer.py"
    if tokenizer_path.exists():
        text = tokenizer_path.read_text(encoding="utf-8", errors="ignore")
        tokens.update(re.findall(r"<[^>\s]+>", text))
    return sorted(tokens)


def combined_reference_tokens(sfm_dir: str | Path | None = None, unigenx_dir: str | Path | None = None) -> list[str]:
    tokens: set[str] = set()
    if sfm_dir:
        tokens.update(naturelm_tokens_from_sfm(sfm_dir))
    if unigenx_dir:
        tokens.update(unigenx_tokens_from_repo(unigenx_dir))
    return sorted(tokens)


def write_tokens(tokens: Iterable[str], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return len(unique)

