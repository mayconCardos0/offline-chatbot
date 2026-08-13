"""
Tests for scripts/generate_eval_dataset.py.

The LLM is always a mocked `llm_chat` callable here — the pure
question-generation/sampling logic is designed to be testable without a real
GGUF model, matching the project's existing convention of mocking
LocalModel/Retriever in unit tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_eval_dataset import (  # noqa: E402
    _NEGATIVE_QUERIES,
    build_dataset_entry,
    build_multi_chunk_question_messages,
    build_negative_entries,
    build_question_generation_messages,
    clean_generated_question,
    generate_dataset,
    generate_question_for_chunk,
    generate_question_for_group,
    sample_chunk_groups,
)

# ---------------------------------------------------------------------------
# clean_generated_question
# ---------------------------------------------------------------------------


class TestCleanGeneratedQuestion:
    def test_accepts_plain_question(self):
        assert (
            clean_generated_question("Qual foi a capital do Império Romano?")
            == "Qual foi a capital do Império Romano?"
        )

    def test_strips_think_blocks(self):
        raw = "<think>raciocínio interno</think>Quem foi Júlio César?"
        assert clean_generated_question(raw) == "Quem foi Júlio César?"

    def test_strips_pergunta_prefix(self):
        raw = "Pergunta: O que foi o Neolítico?"
        assert clean_generated_question(raw) == "O que foi o Neolítico?"

    def test_strips_surrounding_quotes(self):
        raw = '"O que foi a Revolução Industrial?"'
        assert clean_generated_question(raw) == "O que foi a Revolução Industrial?"

    def test_takes_only_first_non_empty_line(self):
        raw = "\n\nQual foi o primeiro imperador romano?\nSegunda linha ignorada."
        assert clean_generated_question(raw) == "Qual foi o primeiro imperador romano?"

    def test_rejects_empty_string(self):
        assert clean_generated_question("") is None

    def test_rejects_whitespace_only(self):
        assert clean_generated_question("   \n  ") is None

    def test_rejects_text_without_question_mark(self):
        assert clean_generated_question("Isso não é uma pergunta.") is None

    def test_rejects_too_short(self):
        assert clean_generated_question("Oi?") is None


# ---------------------------------------------------------------------------
# build_question_generation_messages / generate_question_for_chunk (1 chunk)
# ---------------------------------------------------------------------------


class TestSingleChunkQuestionGeneration:
    def test_messages_include_chunk_text_and_system_prompt(self):
        messages = build_question_generation_messages("O Paleolítico durou...")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "O Paleolítico durou..." in messages[1]["content"]

    def test_generate_question_uses_injected_llm_chat(self):
        calls = []

        def fake_llm_chat(messages):
            calls.append(messages)
            return "O que caracterizou o período Paleolítico?"

        result = generate_question_for_chunk("texto do chunk", fake_llm_chat)

        assert result == "O que caracterizou o período Paleolítico?"
        assert len(calls) == 1
        assert "texto do chunk" in calls[0][1]["content"]

    def test_generate_question_returns_none_for_unusable_output(self):
        result = generate_question_for_chunk("texto", lambda messages: "sem sentido.")
        assert result is None


# ---------------------------------------------------------------------------
# build_multi_chunk_question_messages / generate_question_for_group (2+ chunks)
# ---------------------------------------------------------------------------


class TestMultiChunkQuestionGeneration:
    def test_messages_include_all_chunk_texts_labelled(self):
        messages = build_multi_chunk_question_messages(
            ["texto do Paleolítico...", "texto do Neolítico..."]
        )
        assert messages[0]["role"] == "system"
        assert "TRECHO 1" in messages[1]["content"]
        assert "texto do Paleolítico..." in messages[1]["content"]
        assert "TRECHO 2" in messages[1]["content"]
        assert "texto do Neolítico..." in messages[1]["content"]

    def test_multi_chunk_system_prompt_requires_combining_all_excerpts(self):
        messages = build_multi_chunk_question_messages(["a", "b"])
        assert "TODOS os trechos" in messages[0]["content"]

    def test_generate_question_for_group_uses_single_chunk_prompt_for_one(self):
        calls = []

        def fake_llm_chat(messages):
            calls.append(messages)
            return "O que caracterizou esse trecho único?"

        result = generate_question_for_group(["texto único"], fake_llm_chat)

        assert result == "O que caracterizou esse trecho único?"
        assert "TRECHO:" in calls[0][1]["content"]

    def test_generate_question_for_group_uses_multi_chunk_prompt_for_two_plus(self):
        calls = []

        def fake_llm_chat(messages):
            calls.append(messages)
            return "O que esses dois trechos têm em comum?"

        result = generate_question_for_group(["texto A", "texto B"], fake_llm_chat)

        assert result == "O que esses dois trechos têm em comum?"
        assert "TRECHO 1" in calls[0][1]["content"]
        assert "TRECHO 2" in calls[0][1]["content"]

    def test_generate_question_for_group_returns_none_for_unusable_output(self):
        result = generate_question_for_group(
            ["a", "b"], lambda messages: "sem sentido."
        )
        assert result is None


# ---------------------------------------------------------------------------
# sample_chunk_groups
# ---------------------------------------------------------------------------


def _make_chunks(n, unit="U1", chapter="C1", text_len=300, offset=0):
    return [
        {
            "chunk_id": f"c{offset + i}",
            "text": "x" * text_len,
            "unit": unit,
            "chapter": chapter,
        }
        for i in range(n)
    ]


class TestSampleChunkGroups:
    def test_covers_every_chapter_before_repeating(self):
        chunks = (
            _make_chunks(5, unit="U1", chapter="C1")
            + _make_chunks(5, unit="U2", chapter="C1", offset=5)
            + _make_chunks(5, unit="U3", chapter="C1", offset=10)
        )
        groups = sample_chunk_groups(chunks, group_size=1, n_groups=3, seed=1)
        units = {g[0]["unit"] for g in groups}
        assert len(groups) == 3
        assert units == {"U1", "U2", "U3"}

    def test_respects_n_groups(self):
        chunks = _make_chunks(50)
        groups = sample_chunk_groups(chunks, group_size=1, n_groups=10, seed=1)
        assert len(groups) == 10

    def test_group_size_two_returns_pairs_from_same_chapter(self):
        chunks = _make_chunks(10, unit="U1", chapter="C1")
        groups = sample_chunk_groups(chunks, group_size=2, n_groups=3, seed=1)
        assert len(groups) == 3
        for group in groups:
            assert len(group) == 2
            assert group[0]["chapter"] == group[1]["chapter"]
            # Chunks distintos dentro do grupo.
            assert group[0]["chunk_id"] != group[1]["chunk_id"]

    def test_group_size_three_returns_triples_from_same_chapter(self):
        chunks = _make_chunks(12, unit="U1", chapter="C1")
        groups = sample_chunk_groups(chunks, group_size=3, n_groups=2, seed=1)
        assert len(groups) == 2
        for group in groups:
            assert len(group) == 3
            assert len({c["chunk_id"] for c in group}) == 3

    def test_chapter_with_too_few_chunks_is_skipped_for_larger_groups(self):
        # Só 2 chunks nesse capítulo — não dá para formar um grupo de 3.
        chunks = _make_chunks(2, unit="U1", chapter="C1")
        groups = sample_chunk_groups(chunks, group_size=3, n_groups=5, seed=1)
        assert groups == []

    def test_ignores_chunks_too_short(self):
        chunks = _make_chunks(5, text_len=50)  # abaixo do mínimo de 200 chars
        groups = sample_chunk_groups(chunks, group_size=1, n_groups=5, seed=1)
        assert groups == []

    def test_empty_input_returns_empty(self):
        assert sample_chunk_groups([], group_size=1, n_groups=10, seed=1) == []

    def test_deterministic_with_same_seed(self):
        chunks = _make_chunks(30, unit="U1") + _make_chunks(30, unit="U2", offset=30)
        a = sample_chunk_groups(chunks, group_size=1, n_groups=10, seed=7)
        b = sample_chunk_groups(chunks, group_size=1, n_groups=10, seed=7)
        assert [g[0]["chunk_id"] for g in a] == [g[0]["chunk_id"] for g in b]


# ---------------------------------------------------------------------------
# build_dataset_entry / build_negative_entries — schema compatibility
# ---------------------------------------------------------------------------


class TestDatasetEntrySchema:
    def test_single_chunk_entry_matches_curated_schema(self):
        chunk = {"chunk_id": "abc123", "breadcrumb": "Unidade 1 > Capítulo 1"}
        entry = build_dataset_entry("q0000", "Pergunta gerada?", [chunk])

        assert entry["query"] == "Pergunta gerada?"
        assert entry["relevant_chunks"] == [{"chunk_id": "abc123", "relevance": 3}]
        assert entry["answerable"] is True
        assert entry["category"] == "factual"
        assert entry["n_source_chunks"] == 1

    def test_multi_chunk_entry_lists_every_chunk_as_relevant(self):
        chunks = [
            {"chunk_id": "abc123", "breadcrumb": "Unidade 1 > Capítulo 1 > A"},
            {"chunk_id": "def456", "breadcrumb": "Unidade 1 > Capítulo 1 > B"},
        ]
        entry = build_dataset_entry("q0001", "Pergunta multi-hop?", chunks)

        assert entry["relevant_chunks"] == [
            {"chunk_id": "abc123", "relevance": 3},
            {"chunk_id": "def456", "relevance": 3},
        ]
        assert entry["category"] == "multi_hop"
        assert entry["n_source_chunks"] == 2

    def test_explicit_category_overrides_inference(self):
        entry = build_dataset_entry(
            "q0002", "Pergunta?", [{"chunk_id": "x"}], category="paraphrase"
        )
        assert entry["category"] == "paraphrase"

    def test_negative_entries_have_no_relevant_chunks_and_are_unanswerable(self):
        entries = build_negative_entries(start_index=5)
        assert len(entries) == len(_NEGATIVE_QUERIES)
        for entry in entries:
            assert entry["relevant_chunks"] == []
            assert entry["answerable"] is False
            assert entry["category"] == "negative"


# ---------------------------------------------------------------------------
# generate_dataset — end to end with a mocked LLM
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    def test_generates_single_pair_and_triple_questions_plus_negatives(self):
        chunks = _make_chunks(30, unit="U1", chapter="C1")

        def fake_llm_chat(messages):
            return "Pergunta gerada sobre o(s) trecho(s)?"

        dataset = generate_dataset(
            chunks,
            fake_llm_chat,
            n_single=3,
            n_pairs=2,
            n_triples=1,
            seed=1,
            include_negatives=True,
        )

        factual = [d for d in dataset if d["category"] == "factual"]
        multi_hop = [d for d in dataset if d["category"] == "multi_hop"]
        negative = [d for d in dataset if d["category"] == "negative"]

        assert len(factual) == 3
        assert len(multi_hop) == 3  # 2 pares + 1 trio
        assert len(negative) == len(_NEGATIVE_QUERIES)
        assert all(d["n_source_chunks"] == 1 for d in factual)
        # Dos multi_hop, deve haver tanto pares (2) quanto trios (3).
        assert {d["n_source_chunks"] for d in multi_hop} == {2, 3}

    def test_can_exclude_negatives(self):
        chunks = _make_chunks(5)
        dataset = generate_dataset(
            chunks,
            lambda messages: "Pergunta válida sobre o trecho?",
            n_single=3,
            n_pairs=0,
            n_triples=0,
            include_negatives=False,
        )
        assert all(d["category"] != "negative" for d in dataset)

    def test_skips_groups_whose_llm_output_is_unusable(self):
        chunks = _make_chunks(3)
        dataset = generate_dataset(
            chunks,
            lambda messages: "resposta sem interrogação",
            n_single=3,
            n_pairs=0,
            n_triples=0,
            include_negatives=False,
        )
        assert dataset == []

    def test_zero_pairs_and_triples_generates_only_single_chunk_questions(self):
        chunks = _make_chunks(10)
        dataset = generate_dataset(
            chunks,
            lambda messages: "Isso é uma pergunta válida?",
            n_single=5,
            n_pairs=0,
            n_triples=0,
            include_negatives=False,
        )
        assert len(dataset) == 5
        assert all(d["category"] == "factual" for d in dataset)
