from pathlib import Path

from iska_reasoner.data.dataset import KIND_TO_ID, RandomOrderCollator
from iska_reasoner.data.graphify import graphify_hebrew_row
from iska_reasoner.data.hebrew import (
    graphify_conllu_sentence,
    graphify_verb_complement_row,
    infer_hebrew_root,
    iter_conllu_sentences,
    root_extension_example,
    strip_hebrew_diacritics,
    template_signature,
)
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.tools.verifiers import hebrew_metrics_for_example


def test_hebrew_root_and_template_heuristics():
    assert strip_hebrew_diacritics("כָּתַב") == "כתב"
    assert infer_hebrew_root("הכתבתי") == "כתב"
    assert template_signature("הכתבתי", "כתב")


def test_hebrew_instruction_graph_has_roots_and_answer():
    ex = graphify_hebrew_row(
        {"instruction": "כתוב משפט קצר", "input": "", "output": "שלום עולם"},
        idx=0,
        dataset_name="hebrew_alpaca_train",
    )
    assert ex.task == "hebrew_instruction_sft"
    assert any(node.type == "hebrew_root" for node in ex.nodes)
    assert any(token.startswith("ANSWER:") for token in ex.target_tokens)
    metrics = hebrew_metrics_for_example(ex)
    assert metrics["hebrew/root_node_count_mean"] > 0
    assert metrics["hebrew/qa_or_instruction_rate"] == 1.0


def test_conllu_graphification_adds_binyan_and_dependencies(tmp_path: Path):
    conllu = tmp_path / "sample.conllu"
    conllu.write_text(
        "# sent_id = 1\n"
        "# text = הילד כתב מכתב\n"
        "1\tהילד\tילד\tNOUN\tNOUN\tGender=Masc|Number=Sing\t2\tnsubj\t_\t_\n"
        "2\tכתב\tכתב\tVERB\tVERB\tHebBinyan=PAAL|Tense=Past\t0\troot\t_\t_\n"
        "3\tמכתב\tמכתב\tNOUN\tNOUN\tGender=Masc|Number=Sing\t2\tobj\t_\t_\n\n",
        encoding="utf-8",
    )
    sentence = next(iter_conllu_sentences(conllu))
    ex = graphify_conllu_sentence(sentence, 0, "hebrew_ud_htb")
    assert ex.task == "hebrew_morphosyntax"
    assert any(node.type == "hebrew_binyan" and node.value == "PAAL" for node in ex.nodes)
    assert any(edge.type == "dep:obj" for edge in ex.edges)
    assert any(token == "HEBREW:binyan:PAAL" for token in ex.target_tokens)


def test_verb_complement_schema_support():
    ex = graphify_verb_complement_row(
        {
            "verb_LexiconItem": "אבד",
            "verb_dottedLexiconItem": "אָבַד",
            "verb_binyan": "Pa'al",
            "verb_root": "אבד",
            "complement_LexiconItem": "את",
        },
        idx=0,
    )
    assert ex.task == "hebrew_root_complements"
    assert any(node.type == "verb_complement" for node in ex.nodes)
    assert "HEBREW:binyan:Pa'al" in ex.target_tokens


def test_root_extension_graph_and_gflownet_targets():
    ex = root_extension_example("כתב", "write", 0)
    assert ex.task == "hebrew_root_extension"
    assert any(node.type == "hebrew_derived_form" for node in ex.nodes)
    assert any(token.startswith("HEBREW:derived:") for token in ex.target_tokens)
    metrics = hebrew_metrics_for_example(ex)
    assert metrics["hebrew/derived_form_count_mean"] >= 5


def test_collator_reserves_target_positions_for_long_hebrew_source():
    ex = graphify_hebrew_row({"text": " ".join(["כתב"] * 300)}, idx=1, dataset_name="hebrew_sefaria_train")
    vocab = build_vocab([ex])
    collator = RandomOrderCollator(vocab=vocab, max_source_tokens=220, max_target_tokens=16, max_seq_len=64, order_mode="first")
    batch = collator([ex])
    pos_mask = batch["kind_ids"].eq(KIND_TO_ID["position"])
    assert pos_mask.any()
    assert batch["labels"][pos_mask].ne(-100).all()
