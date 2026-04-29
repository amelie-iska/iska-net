from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import Edge, GraphExample, Node

HEBREW_LETTER_RE = re.compile(r"[\u05d0-\u05ea]+")
FINAL_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
COMMON_PREFIXES = tuple("והבכלמש")
COMMON_SUFFIXES = ("ים", "ות", "נו", "כם", "כן", "יה", "יו", "תי", "תה", "תם", "תן", "ה")
WEAK_LETTERS = set("אהוי")

ROOT_SEEDS = [
    ("כתב", "write"),
    ("למד", "learn"),
    ("שמר", "guard"),
    ("אמר", "say"),
    ("הלך", "walk"),
    ("גדל", "grow"),
    ("פעל", "act"),
    ("אכל", "eat"),
    ("ראה", "see"),
    ("נתן", "give"),
    ("קבל", "receive"),
    ("דבר", "speak"),
    ("חשב", "think"),
    ("רפא", "heal"),
    ("בדק", "check"),
    ("זכר", "remember"),
]

BINYAN_TEMPLATES = {
    "paal": "{r1}{r2}{r3}",
    "piel": "{r1}{r2}{r3}",
    "hifil": "ה{r1}{r2}י{r3}",
    "hitpael": "הת{r1}{r2}{r3}",
    "nifal": "נ{r1}{r2}{r3}",
}


def strip_hebrew_diacritics(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("M"))


def normalize_hebrew(text: str) -> str:
    return strip_hebrew_diacritics(text).translate(FINAL_MAP)


def hebrew_tokens(text: str, max_tokens: int = 96) -> list[str]:
    return HEBREW_LETTER_RE.findall(normalize_hebrew(text))[:max_tokens]


def infer_hebrew_root(word: str) -> str:
    clean = normalize_hebrew(word)
    letters = "".join(HEBREW_LETTER_RE.findall(clean))
    if not letters:
        return ""
    if len(letters) > 3 and letters.startswith(COMMON_PREFIXES):
        letters = letters[1:]
    for suffix in COMMON_SUFFIXES:
        if len(letters) - len(suffix) >= 3 and letters.endswith(suffix):
            letters = letters[: -len(suffix)]
            break
    consonants = [ch for ch in letters if ch not in WEAK_LETTERS]
    if len(consonants) >= 3:
        return "".join(consonants[:3])
    return letters[:3] if len(letters) >= 3 else letters


def template_signature(word: str, root: str) -> str:
    clean = normalize_hebrew(word)
    if not clean or not root:
        return ""
    slots = {letter: f"C{i + 1}" for i, letter in enumerate(root[:4])}
    used: dict[str, int] = {}
    out: list[str] = []
    for ch in clean:
        if ch in slots:
            used[ch] = used.get(ch, 0) + 1
            out.append(slots[ch] if used[ch] == 1 else slots[ch] + "'")
        elif "\u05d0" <= ch <= "\u05ea":
            out.append("V")
        else:
            out.append(ch)
    return "".join(out)[:32]


def _add_root_nodes(nodes: list[Node], edges: list[Edge], anchor: str, word: str, root: str, prefix: str) -> None:
    if not root:
        return
    root_id = f"{prefix}_root"
    template_id = f"{prefix}_template"
    nodes.append(Node(id=root_id, type="hebrew_root", value=root, features={"heuristic": True}))
    nodes.append(Node(id=template_id, type="hebrew_template", value=template_signature(word, root)))
    edges.append(Edge(src=anchor, dst=root_id, type="has_shoresh"))
    edges.append(Edge(src=anchor, dst=template_id, type="has_template"))
    edges.append(Edge(src=root_id, dst=template_id, type="fills_template"))
    for idx, radical in enumerate(root):
        rid = f"{prefix}_rad{idx}"
        nodes.append(Node(id=rid, type="hebrew_radical", value=radical, features={"position": idx + 1}))
        edges.append(Edge(src=root_id, dst=rid, type="has_radical"))


def hebrew_text_graph(
    text: str,
    idx: int,
    dataset_name: str,
    task: str = "hebrew_morphology",
    extra_nodes: list[Node] | None = None,
    extra_edges: list[Edge] | None = None,
    metadata: dict[str, Any] | None = None,
    answer: str | None = None,
) -> GraphExample:
    tokens = hebrew_tokens(text)
    nodes: list[Node] = [Node(id="text", type="hebrew_text", value=text[:2048])]
    edges: list[Edge] = []
    roots: list[str] = []
    templates: list[str] = []
    prev = None
    for i, token in enumerate(tokens[:64]):
        tid = f"tok{i}"
        nodes.append(Node(id=tid, type="hebrew_token", value=token))
        edges.append(Edge(src="text", dst=tid, type="contains_token"))
        if prev is not None:
            edges.append(Edge(src=prev, dst=tid, type="next_token"))
        prev = tid
        root = infer_hebrew_root(token)
        if root:
            roots.append(root)
            templates.append(template_signature(token, root))
            _add_root_nodes(nodes, edges, tid, token, root, f"tok{i}")
    if extra_nodes:
        nodes.extend(extra_nodes)
    if extra_edges:
        edges.extend(extra_edges)
    target_tokens = ["HEBREW:task:morphology"]
    target_tokens.extend(f"HEBREW:root:{root}" for root in sorted(set(roots))[:8])
    target_tokens.extend(f"HEBREW:template:{tmpl}" for tmpl in sorted(set(t for t in templates if t))[:4])
    if answer:
        target_tokens.append(f"ANSWER:{answer[:120]}")
    stable = hashlib.sha1(f"{dataset_name}\t{idx}\t{text[:160]}".encode("utf-8")).hexdigest()[:12]
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{stable}",
        task=task,
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens or ["HEBREW:task:morphology"],
        metadata={"source_dataset": dataset_name, "roots": sorted(set(roots))[:32], **(metadata or {})},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def parse_feats(feats: str) -> dict[str, str]:
    if not feats or feats == "_":
        return {}
    out = {}
    for item in feats.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def iter_conllu_sentences(path: str | Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    comments: dict[str, str] = {}
    tokens: list[dict[str, Any]] = []
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    yield {"comments": comments, "tokens": tokens}
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                comments = {}
                tokens = []
                continue
            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    comments[key.strip()] = value.strip()
                continue
            parts = line.split("\t")
            if len(parts) != 10 or "-" in parts[0] or "." in parts[0]:
                continue
            feats = parse_feats(parts[5])
            tokens.append(
                {
                    "id": int(parts[0]),
                    "form": parts[1],
                    "lemma": parts[2] if parts[2] != "_" else "",
                    "upos": parts[3],
                    "xpos": parts[4],
                    "feats": feats,
                    "head": int(parts[6]) if parts[6].isdigit() else 0,
                    "deprel": parts[7],
                    "misc": parts[9],
                }
            )
    if tokens and (limit is None or yielded < limit):
        yield {"comments": comments, "tokens": tokens}


def graphify_conllu_sentence(sentence: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    text = sentence.get("comments", {}).get("text") or " ".join(tok["form"] for tok in sentence["tokens"])
    nodes = [Node(id="sent", type="hebrew_sentence", value=text[:2048])]
    edges: list[Edge] = []
    roots: list[str] = []
    binyans: list[str] = []
    for tok in sentence["tokens"][:96]:
        tid = f"tok{tok['id']}"
        lemma = tok.get("lemma") or tok["form"]
        root = infer_hebrew_root(lemma or tok["form"])
        nodes.append(
            Node(
                id=tid,
                type="hebrew_token",
                value=tok["form"],
                features={"upos": tok["upos"], "xpos": tok["xpos"], "lemma": lemma, **tok.get("feats", {})},
            )
        )
        edges.append(Edge(src="sent", dst=tid, type="contains_token"))
        if lemma:
            lid = f"lemma{tok['id']}"
            nodes.append(Node(id=lid, type="hebrew_lemma", value=lemma))
            edges.append(Edge(src=tid, dst=lid, type="has_lemma"))
        if root:
            roots.append(root)
            _add_root_nodes(nodes, edges, tid, tok["form"], root, f"tok{tok['id']}")
        binyan = tok.get("feats", {}).get("HebBinyan")
        if binyan:
            bid = f"binyan{tok['id']}"
            binyans.append(binyan)
            nodes.append(Node(id=bid, type="hebrew_binyan", value=binyan))
            edges.append(Edge(src=tid, dst=bid, type="has_binyan"))
    token_ids = {tok["id"] for tok in sentence["tokens"]}
    for tok in sentence["tokens"][:96]:
        if tok["head"] in token_ids:
            edges.append(Edge(src=f"tok{tok['head']}", dst=f"tok{tok['id']}", type=f"dep:{tok['deprel']}"))
    target_tokens = ["HEBREW:task:ud_morphosyntax"]
    target_tokens.extend(f"HEBREW:root:{root}" for root in sorted(set(roots))[:8])
    target_tokens.extend(f"HEBREW:binyan:{binyan}" for binyan in sorted(set(binyans))[:4])
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{sentence.get('comments', {}).get('sent_id', idx)}",
        task="hebrew_morphosyntax",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name, "roots": sorted(set(roots)), "binyans": sorted(set(binyans))},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def root_extension_example(root: str, gloss: str, idx: int, dataset_name: str = "hebrew_root_synthetic") -> GraphExample:
    radicals = list(normalize_hebrew(root)[:3])
    nodes = [Node(id="root", type="hebrew_root", value=root, features={"gloss": gloss})]
    edges: list[Edge] = []
    for i, radical in enumerate(radicals):
        rid = f"rad{i}"
        nodes.append(Node(id=rid, type="hebrew_radical", value=radical, features={"position": i + 1}))
        edges.append(Edge(src="root", dst=rid, type="has_radical"))
    derived_tokens = []
    for name, pattern in BINYAN_TEMPLATES.items():
        if len(radicals) < 3:
            continue
        form = pattern.format(r1=radicals[0], r2=radicals[1], r3=radicals[2])
        bid = f"binyan_{name}"
        fid = f"form_{name}"
        nodes.append(Node(id=bid, type="hebrew_binyan", value=name))
        nodes.append(Node(id=fid, type="hebrew_derived_form", value=form))
        edges.append(Edge(src="root", dst=bid, type="extends_with_binyan"))
        edges.append(Edge(src=bid, dst=fid, type="generates_form"))
        edges.append(Edge(src="root", dst=fid, type="same_shoresh_family"))
        derived_tokens.append(f"HEBREW:derived:{form}")
    target_tokens = [f"HEBREW:root:{root}"] + [f"HEBREW:radical:{r}" for r in radicals] + derived_tokens[:5]
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{root}",
        task="hebrew_root_extension",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name, "root": root, "gloss": gloss, "derived_count": len(derived_tokens)},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def iter_synthetic_root_examples(count: int = 64) -> Iterable[GraphExample]:
    for idx in range(count):
        root, gloss = ROOT_SEEDS[idx % len(ROOT_SEEDS)]
        yield root_extension_example(root, gloss, idx)


def graphify_verb_complement_row(row: dict[str, Any], idx: int, dataset_name: str = "hebrew_verb_complements_lexicon") -> GraphExample:
    verb = str(row.get("verb_LexiconItem") or row.get("verb") or "")
    dotted = str(row.get("verb_dottedLexiconItem") or "")
    root = normalize_hebrew(str(row.get("verb_root") or infer_hebrew_root(verb)))
    binyan = str(row.get("verb_binyan") or "")
    complement = str(row.get("complement_LexiconItem") or row.get("verb_complement") or "")
    nodes = [
        Node(id="verb", type="hebrew_verb", value=verb),
        Node(id="root", type="hebrew_root", value=root),
        Node(id="complement", type="verb_complement", value=complement),
    ]
    edges = [Edge(src="verb", dst="root", type="has_shoresh"), Edge(src="verb", dst="complement", type="selects_complement")]
    if dotted:
        nodes.append(Node(id="dotted", type="hebrew_diacritized", value=dotted))
        edges.append(Edge(src="verb", dst="dotted", type="vocalized_as"))
    if binyan:
        nodes.append(Node(id="binyan", type="hebrew_binyan", value=binyan))
        edges.append(Edge(src="verb", dst="binyan", type="has_binyan"))
    target_tokens = [f"HEBREW:root:{root}", f"HEBREW:verb:{verb}"]
    if binyan:
        target_tokens.append(f"HEBREW:binyan:{binyan}")
    if complement:
        target_tokens.append(f"HEBREW:complement:{complement[:80]}")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{root}_{verb}",
        task="hebrew_root_complements",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name, "root": root, "binyan": binyan},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def iter_verb_complement_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        yield from csv.DictReader(f, dialect=dialect)
