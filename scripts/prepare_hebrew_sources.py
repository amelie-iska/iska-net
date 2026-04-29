#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.hebrew import (  # noqa: E402
    graphify_conllu_sentence,
    hebrew_text_graph,
    iter_conllu_sentences,
    iter_synthetic_root_examples,
    iter_verb_complement_rows,
    graphify_verb_complement_row,
    strip_hebrew_diacritics,
)
from iska_reasoner.graph.orders import build_orders  # noqa: E402
from iska_reasoner.graph.schema import Edge, GraphExample, Node  # noqa: E402
from iska_reasoner.utils.io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_json_or_jsonl(path: Path) -> Iterable[Any]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
    elif path.suffix == ".json":
        yield _read_json(path)


def graphify_hebrew_qa_record(context: str, question: str, answers: list[str], idx: int, dataset_name: str) -> GraphExample:
    answer = answers[0] if answers else ""
    extra_nodes = [
        Node(id="context", type="hebrew_context", value=context[:2048]),
        Node(id="question", type="hebrew_question", value=question[:1024]),
        Node(id="answer", type="hebrew_answer", value=answer[:512]),
    ]
    extra_edges = [Edge(src="context", dst="question", type="asked_about"), Edge(src="question", dst="answer", type="answered_by")]
    ex = hebrew_text_graph(
        "\n".join([context, question, answer]),
        idx,
        dataset_name,
        task="hebrew_question_answering",
        extra_nodes=extra_nodes,
        extra_edges=extra_edges,
        metadata={"answer_count": len(answers)},
        answer=answer,
    )
    ex.target_tokens.append("HEBREW:task:qa")
    if answer:
        ex.target_tokens.append(f"ANSWER:{answer[:120]}")
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def iter_heq_examples(repo_dir: Path, limit: int) -> Iterable[GraphExample]:
    idx = 0
    for path in sorted(repo_dir.rglob("*.json")) + sorted(repo_dir.rglob("*.jsonl")):
        if ".git" in path.parts:
            continue
        for payload in _iter_json_or_jsonl(path):
            records = payload.get("data", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                continue
            for article in records:
                paragraphs = article.get("paragraphs", []) if isinstance(article, dict) else []
                for paragraph in paragraphs:
                    context = str(paragraph.get("context") or paragraph.get("paragraph") or "")
                    for qa in paragraph.get("qas", []):
                        question = str(qa.get("question") or "")
                        raw_answers = qa.get("answers") or []
                        answers = []
                        for ans in raw_answers:
                            if isinstance(ans, dict):
                                answers.append(str(ans.get("text") or ""))
                            else:
                                answers.append(str(ans))
                        yield graphify_hebrew_qa_record(context, question, [a for a in answers if a], idx, "hebrew_qa_nnlp")
                        idx += 1
                        if idx >= limit:
                            return


def iter_nakdimon_examples(repo_dir: Path, limit: int) -> Iterable[GraphExample]:
    idx = 0
    candidates = []
    for pattern in ("*.txt", "*.he", "*.text"):
        candidates.extend(repo_dir.rglob(pattern))
    for path in sorted(candidates):
        if ".git" in path.parts or path.stat().st_size > 200_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")[:4000]
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        undotted = strip_hebrew_diacritics(text)
        extra_nodes = [Node(id="dotted", type="hebrew_diacritized", value=text[:2048])]
        extra_edges = [Edge(src="text", dst="dotted", type="vocalized_as")]
        ex = hebrew_text_graph(
            undotted,
            idx,
            "hebrew_nakdimon",
            task="hebrew_diacritization",
            extra_nodes=extra_nodes,
            extra_edges=extra_edges,
            metadata={"source_file": str(path.relative_to(repo_dir))},
            answer=text[:120],
        )
        ex.target_tokens.append("HEBREW:task:diacritization")
        ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
        yield ex
        idx += 1
        if idx >= limit:
            return


def write_ud(raw_dir: Path, out_dir: Path, limit: int) -> int:
    repo = raw_dir / "hebrew_ud_htb" / "repo"
    files = [repo / "he_htb-ud-train.conllu", repo / "he_htb-ud-dev.conllu", repo / "he_htb-ud-test.conllu"]
    examples = []
    idx = 0
    for path in files:
        if not path.exists():
            continue
        for sent in iter_conllu_sentences(path, limit=max(0, limit - idx)):
            examples.append(graphify_conllu_sentence(sent, idx, "hebrew_ud_htb"))
            idx += 1
            if idx >= limit:
                break
        if idx >= limit:
            break
    return write_jsonl(ensure_dir(out_dir / "hebrew_ud_htb") / "train.jsonl", (ex.to_dict() for ex in examples))


def write_heq(raw_dir: Path, out_dir: Path, limit: int) -> int:
    repo = raw_dir / "hebrew_qa_nnlp" / "repo"
    return write_jsonl(
        ensure_dir(out_dir / "hebrew_qa_nnlp") / "train.jsonl",
        (ex.to_dict() for ex in tqdm(iter_heq_examples(repo, limit), desc="graphify/hebrew_qa_nnlp", total=limit)),
    )


def write_nakdimon(raw_dir: Path, out_dir: Path, limit: int) -> int:
    repo = raw_dir / "hebrew_nakdimon" / "repo"
    return write_jsonl(
        ensure_dir(out_dir / "hebrew_nakdimon") / "train.jsonl",
        (ex.to_dict() for ex in tqdm(iter_nakdimon_examples(repo, limit), desc="graphify/hebrew_nakdimon", total=limit)),
    )


def write_roots(out_dir: Path, count: int) -> int:
    return write_jsonl(
        ensure_dir(out_dir / "hebrew_root_synthetic") / "train.jsonl",
        (ex.to_dict() for ex in tqdm(iter_synthetic_root_examples(count), desc="graphify/hebrew_root_synthetic", total=count)),
    )


def write_verb_complements(raw_dir: Path, out_dir: Path, limit: int) -> int:
    base = raw_dir / "hebrew_verb_complements_lexicon"
    files = list(base.glob("*.tsv")) + list(base.glob("*.csv"))
    if not files:
        target = ensure_dir(out_dir / "hebrew_verb_complements_lexicon") / "train.jsonl"
        return write_jsonl(target, [])
    rows = []
    idx = 0
    for path in files:
        for row in iter_verb_complement_rows(path):
            rows.append(graphify_verb_complement_row(row, idx).to_dict())
            idx += 1
            if idx >= limit:
                break
        if idx >= limit:
            break
    return write_jsonl(ensure_dir(out_dir / "hebrew_verb_complements_lexicon") / "train.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hebrew GitHub/local sources and root-extension examples.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--ud-limit", type=int, default=64)
    parser.add_argument("--qa-limit", type=int, default=64)
    parser.add_argument("--nakdimon-limit", type=int, default=32)
    parser.add_argument("--root-count", type=int, default=64)
    parser.add_argument("--verb-complements-limit", type=int, default=64)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    counts = {
        "hebrew_ud_htb": write_ud(raw_dir, out_dir, args.ud_limit),
        "hebrew_qa_nnlp": write_heq(raw_dir, out_dir, args.qa_limit),
        "hebrew_nakdimon": write_nakdimon(raw_dir, out_dir, args.nakdimon_limit),
        "hebrew_root_synthetic": write_roots(out_dir, args.root_count),
        "hebrew_verb_complements_lexicon": write_verb_complements(raw_dir, out_dir, args.verb_complements_limit),
    }
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
