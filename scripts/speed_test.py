"""
Speed & accuracy benchmark — runs ALL GGUF models in models/ and compares results.

Modes
─────
  Default (latency + accuracy):
      python scripts/speed_test.py

  Generation quality benchmark (RAG-focused metrics):
      python scripts/speed_test.py --benchmark-generation
      python scripts/speed_test.py --benchmark-generation --model models/gemma.gguf

  Combined (run both):
      python scripts/speed_test.py --benchmark-generation --all

  Custom dataset for generation benchmark:
      python scripts/speed_test.py --benchmark-generation --gen-dataset data/eval/gen_dataset.json

  Save generation report:
      python scripts/speed_test.py --benchmark-generation --save-gen-report reports/gen_v1.json

Generation metrics
──────────────────
  ROUGE-L          — Longest common subsequence overlap with reference (fluency proxy)
  BERTScore-F1     — Semantic similarity via embeddings (paraphrase-robust)
  Faithfulness     — Fraction of response sentences supported by retrieved context
  Answer Relevance — Cosine similarity between response and query embeddings
  Exact Match      — Binary hit for factual answers (names, dates, formulas)
  No-Context Rate  — How often the pipeline refuses to answer (guardrail sensitivity)
  TTFT             — Time to First Token (ms)
  TPS              — Tokens per Second
  Peak RSS         — Peak memory usage (MB), critical for Raspberry Pi
"""

import argparse
import json
import math
import os
import re

try:
    import resource
except ImportError:
    resource = None
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Q&A pairs (original latency + accuracy benchmark)
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {"prompt": "Quanto é 15 + 27? Responda apenas com o número.", "expected": ["42"]},
    {
        "prompt": "Quanto é 144 dividido por 12? Responda apenas com o número.",
        "expected": ["12"],
    },
    {
        "prompt": "Qual é a raiz quadrada de 81? Responda apenas com o número.",
        "expected": ["9"],
    },
    {
        "prompt": "Quantos números primos existem entre 1 e 10? Responda apenas com o número.",
        "expected": ["4"],
    },
    {"prompt": "Quanto é 7 x 8? Responda apenas com o número.", "expected": ["56"]},
    {
        "prompt": "Qual é a capital da França? Responda apenas com o nome da cidade.",
        "expected": ["Paris"],
    },
    {
        "prompt": "Qual é a capital do Japão? Responda apenas com o nome da cidade.",
        "expected": ["Toquio", "Tóquio"],
    },
    {
        "prompt": "Qual é o maior oceano da Terra? Responda apenas com o nome.",
        "expected": ["Pacifico", "Pacífico"],
    },
    {
        "prompt": "Em qual continente fica o Egito? Responda apenas com o nome do continente.",
        "expected": ["Africa", "África"],
    },
    {
        "prompt": "Qual é o rio mais longo do mundo? Responda apenas com o nome.",
        "expected": ["Nilo", "Amazonas"],
    },
    {
        "prompt": "Qual é o símbolo químico do ouro? Responda apenas com o símbolo.",
        "expected": ["Au"],
    },
    {
        "prompt": "Qual é o símbolo químico da água? Responda apenas com a fórmula.",
        "expected": ["H2O"],
    },
    {
        "prompt": "Quantos planetas existem no nosso sistema solar? Responda apenas com o número.",
        "expected": ["8"],
    },
    {
        "prompt": "Qual é o ponto de ebulição da água em Celsius? Responda apenas com o número.",
        "expected": ["100"],
    },
    {
        "prompt": "Qual gás as plantas absorvem da atmosfera? Responda apenas com o nome.",
        "expected": ["dióxido de carbono", "CO2", "dioxido de carbono"],
    },
    {
        "prompt": "Em que ano terminou a Segunda Guerra Mundial? Responda apenas com o ano.",
        "expected": ["1945"],
    },
    {
        "prompt": "Quem escreveu Romeu e Julieta? Responda apenas com o nome.",
        "expected": ["Shakespeare", "William Shakespeare"],
    },
    {
        "prompt": "Quantos lados tem um hexágono? Responda apenas com o número.",
        "expected": ["6"],
    },
    {
        "prompt": "Qual é o ponto de congelamento da água em Fahrenheit? Responda apenas com o número.",
        "expected": ["32"],
    },
    {
        "prompt": "Quantos dias há em um ano bissexto? Responda apenas com o número.",
        "expected": ["366"],
    },
]

# ---------------------------------------------------------------------------
# Generation benchmark dataset
# Each item has:
#   query       — the user question
#   context     — simulated RAG context (chunks) injected into the system prompt
#   reference   — ground-truth answer for ROUGE-L and BERTScore
#   expected    — list of strings for Exact Match (factual answers)
#   category    — "factual" | "explanation" | "comparison" | "no_context"
# ---------------------------------------------------------------------------
GENERATION_DATASET = [
    # ── Factual / historical (PT-BR focus) ──────────────────────────────
    {
        "query": "Em que ano ocorreu a Proclamação da República no Brasil?",
        "context": (
            "A Proclamação da República do Brasil ocorreu em 15 de novembro de 1889, "
            "quando o Marechal Deodoro da Fonseca liderou um golpe militar que pôs fim "
            "ao Segundo Reinado e ao governo de Dom Pedro II."
        ),
        "reference": (
            "A Proclamação da República no Brasil ocorreu em 15 de novembro de 1889, "
            "liderada pelo Marechal Deodoro da Fonseca."
        ),
        "expected": ["1889"],
        "category": "factual",
    },
    {
        "query": "O que foi a Revolução Industrial e onde ela começou?",
        "context": (
            "A Revolução Industrial foi um período de grandes transformações econômicas e "
            "tecnológicas que teve início na Inglaterra no século XVIII, entre 1760 e 1840. "
            "Marcou a transição de uma economia agrária e artesanal para uma economia "
            "industrial e mecanizada, com o surgimento das fábricas, uso da máquina a vapor "
            "e concentração de trabalhadores em centros urbanos."
        ),
        "reference": (
            "A Revolução Industrial foi um período de transformação econômica e tecnológica "
            "iniciado na Inglaterra no século XVIII, caracterizado pela mecanização da produção "
            "e surgimento das fábricas."
        ),
        "expected": ["Inglaterra", "século XVIII"],
        "category": "explanation",
    },
    {
        "query": "Quais foram as principais características do Estado Novo de Getúlio Vargas?",
        "context": (
            "O Estado Novo foi um regime autoritário instaurado por Getúlio Vargas em 10 de "
            "novembro de 1937, após um golpe que dissolveu o Congresso Nacional. Durou até "
            "1945 e se caracterizou pela centralização do poder, censura à imprensa, "
            "perseguição a opositores, criação da CLT (Consolidação das Leis do Trabalho) "
            "em 1943 e forte apelo ao nacionalismo e ao trabalhismo."
        ),
        "reference": (
            "O Estado Novo (1937-1945) foi caracterizado por centralização do poder nas mãos "
            "de Vargas, censura, repressão política, criação da CLT e forte apelo "
            "nacionalista e trabalhista."
        ),
        "expected": ["1937", "centralização", "CLT"],
        "category": "explanation",
    },
    {
        "query": "Qual a diferença entre fotossíntese e respiração celular?",
        "context": (
            "A fotossíntese é o processo pelo qual plantas, algas e cianobactérias convertem "
            "luz solar, dióxido de carbono e água em glicose e oxigênio, ocorrendo nos "
            "cloroplastos. A equação geral é: 6CO₂ + 6H₂O + luz → C₆H₁₂O₆ + 6O₂. "
            "A respiração celular é o processo inverso: as células decompõem glicose na "
            "presença de oxigênio para liberar energia (ATP), produzindo CO₂ e H₂O como "
            "subprodutos. Ocorre nas mitocôndrias."
        ),
        "reference": (
            "A fotossíntese converte luz em glicose (ocorre nos cloroplastos), consumindo CO₂ "
            "e liberando O₂. A respiração celular é o processo inverso: decompõe glicose para "
            "gerar ATP, consumindo O₂ e liberando CO₂ (ocorre nas mitocôndrias)."
        ),
        "expected": ["cloroplastos", "mitocôndrias", "ATP"],
        "category": "comparison",
    },
    {
        "query": "O que foi o Iluminismo e quais seus principais pensadores?",
        "context": (
            "O Iluminismo foi um movimento intelectual e filosófico que floresceu na Europa "
            "durante o século XVIII, também chamado de 'Século das Luzes'. Pregava o uso da "
            "razão como guia para o conhecimento e a organização da sociedade, em oposição "
            "ao absolutismo e à superstição. Seus principais pensadores incluem Voltaire "
            "(liberdade de expressão e tolerância religiosa), Montesquieu (separação dos "
            "poderes), Rousseau (contrato social e soberania popular) e John Locke "
            "(direitos naturais e governo por consentimento)."
        ),
        "reference": (
            "O Iluminismo foi um movimento filosófico europeu do século XVIII baseado na razão. "
            "Principais pensadores: Voltaire (tolerância e liberdade), Montesquieu (separação "
            "dos poderes), Rousseau (contrato social) e Locke (direitos naturais)."
        ),
        "expected": ["Voltaire", "Montesquieu", "Rousseau", "século XVIII"],
        "category": "explanation",
    },
    {
        "query": "Como funciona a Lei de Gravitação Universal de Newton?",
        "context": (
            "A Lei da Gravitação Universal, formulada por Isaac Newton em 1687 na obra "
            "Principia Mathematica, afirma que dois corpos se atraem com uma força diretamente "
            "proporcional ao produto de suas massas e inversamente proporcional ao quadrado da "
            "distância entre eles. A equação é F = G × (m₁ × m₂) / r², onde G é a constante "
            "gravitacional (6,674 × 10⁻¹¹ N·m²/kg²), m₁ e m₂ são as massas e r é a distância "
            "entre os centros dos corpos."
        ),
        "reference": (
            "A Lei de Gravitação Universal de Newton (1687) afirma que a força de atração entre "
            "dois corpos é proporcional ao produto de suas massas e inversamente proporcional ao "
            "quadrado da distância: F = G × (m₁ × m₂) / r²."
        ),
        "expected": ["Newton", "1687", "massa", "distância"],
        "category": "explanation",
    },
    # ── No-context scenario (guardrail test) ────────────────────────────
    {
        "query": "Qual foi o papel da República de Saló na Segunda Guerra Mundial?",
        "context": (
            "A Segunda Guerra Mundial foi um conflito global entre 1939 e 1945 que envolveu "
            "as principais potências mundiais. As causas incluem o Tratado de Versalhes, "
            "a crise econômica de 1929 e o surgimento do nazismo na Alemanha."
        ),
        "reference": None,  # context does not contain information about Repubblica di Salò
        "expected": [],
        "category": "no_context",
    },
    {
        "query": "Explique o processo de mitose celular.",
        "context": (
            "A mitose é um tipo de divisão celular que resulta em duas células-filhas "
            "geneticamente idênticas à célula-mãe, cada uma com o mesmo número de "
            "cromossomos (diploides). É fundamental para o crescimento, reparação de tecidos "
            "e reprodução assexuada. As fases são: prófase (condensação dos cromossomos), "
            "metáfase (alinhamento na placa equatorial), anáfase (separação das cromátides) "
            "e telófase (formação de duas células)."
        ),
        "reference": (
            "A mitose é a divisão celular que produz duas células-filhas idênticas à célula-mãe. "
            "Suas fases são: prófase, metáfase, anáfase e telófase, resultando em células "
            "diploides geneticamente idênticas."
        ),
        "expected": ["prófase", "metáfase", "anáfase", "telófase"],
        "category": "explanation",
    },
]

REAL_RAG_DATASET = [
    {
        "query": "Como ocorreu a Proclamação da República no Brasil?",
        "context": "",
        "reference": (
            "A Proclamação da República foi conduzida por militares, inaugurou "
            "uma nova ordem política no Brasil e abriu a República da Espada."
        ),
        "expected": ["Deodoro", "república", "militares"],
        "category": "factual",
    },
    {
        "query": "Como foi o Segundo Governo Vargas?",
        "context": "",
        "reference": (
            "O Segundo Governo Vargas (1951-1954) priorizou a expansão industrial, "
            "a relação entre Estado e empresários, a empresa pública e o BNDE para "
            "financiar o Plano Lafer."
        ),
        "expected": ["1951", "1954", "BNDE", "Plano Lafer"],
        "category": "explanation",
    },
    {
        "query": "Quais foram as principais características do Estado Novo?",
        "context": "",
        "reference": (
            "O Estado Novo (1937-1945) concentrou poderes no Executivo, substituiu "
            "governadores por interventores, restringiu direitos e usou órgãos como "
            "o DIP para propaganda oficial."
        ),
        "expected": ["1937", "1945", "DIP", "Executivo"],
        "category": "explanation",
    },
    {
        "query": "O que foi o Iluminismo?",
        "context": "",
        "reference": (
            "O Iluminismo foi um movimento intelectual europeu do século XVIII "
            "associado à valorização da razão, ao liberalismo e a pensadores como "
            "Voltaire, Montesquieu, Rousseau e Locke."
        ),
        "expected": ["razão", "Voltaire", "Montesquieu", "Rousseau"],
        "category": "explanation",
    },
    {
        "query": "Quais mudanças marcaram a Revolução Industrial?",
        "context": "",
        "reference": (
            "A Revolução Industrial envolveu mecanização, produção fabril, novas "
            "técnicas, mudanças no trabalho e transformações econômicas e sociais."
        ),
        "expected": ["máquina", "fábrica", "trabalho"],
        "category": "explanation",
    },
    {
        "query": "Qual foi o papel da República de Saló na Segunda Guerra Mundial?",
        "context": "",
        "reference": None,
        "expected": [],
        "category": "no_context",
    },
]

# System prompt template used for generation benchmark (mirrors pipeline.py)
_SYSTEM_TEMPLATE = """\
Você é um professor particular especializado no Ensino Médio brasileiro.
NÃO inclua raciocínio interno ou textos como <think>. Responda apenas com a resposta final.
Responda SEMPRE em português brasileiro correto.
Use EXCLUSIVAMENTE as informações do CONTEXTO fornecido.
Se o contexto não contiver a informação, responda: "Esse conteúdo não está no material disponível."

CONTEXTO:
{context}
"""

_NO_CONTEXT_MARKER = "não está no material"


# ===========================================================================
# ── SECTION 1: Original latency / accuracy benchmark (unchanged) ───────────
# ===========================================================================


def check_accuracy(response: str, expected: list[str]) -> bool:
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    normalized = response.translate(subscript_map).lower()
    return any(ans.lower() in normalized for ans in expected)


def run_single(llm, prompt: str) -> dict:
    messages = [{"role": "user", "content": prompt}]
    ttft = None
    response_text = ""
    token_count = 0
    t0 = time.perf_counter()

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.5,
        top_p=0.8,
        stream=True,
    )

    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        token = choices[0].get("delta", {}).get("content", "")
        if token:
            if ttft is None:
                ttft = time.perf_counter() - t0
            response_text += token
            token_count += 1

    total_time = time.perf_counter() - t0
    return {
        "ttft": ttft or 0.0,
        "tps": token_count / total_time if total_time > 0 else 0,
        "tokens": token_count,
        "total_s": total_time,
        "response": response_text,
    }


def benchmark_model(
    model_path: Path, n_ctx: int, threads: int, gpu_layers: int
) -> dict | None:
    """Load a model, run all QA pairs, return summary dict."""
    from llama_cpp import Llama

    name = model_path.name
    print(f"\n{'='*75}")
    print(f"  Model: {name}")
    print(f"{'='*75}")

    print("  Loading...", end=" ", flush=True)
    try:
        t_load = time.perf_counter()
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        load_s = time.perf_counter() - t_load
        print(f"done in {load_s:.1f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    sys.stdout.write("  Warming up... ")
    sys.stdout.flush()
    try:
        run_single(llm, "hi")
        print("done")
    except Exception:
        print("skipped")

    print(f"\n  {'#':<4} {'TTFT(ms)':<11} {'TPS':<9} {'OK':<5} RESPONSE")
    print(f"  {'-'*68}")

    total_tps, total_ttft, correct = 0.0, 0.0, 0
    wrong = []

    for i, qa in enumerate(QA_PAIRS):
        sys.stdout.write(f"  Running {i+1}/{len(QA_PAIRS)}...\r")
        sys.stdout.flush()
        try:
            res = run_single(llm, qa["prompt"])
        except Exception as e:
            print(f"  {i+1:<4} ERROR: {e}")
            continue

        ok = check_accuracy(res["response"], qa["expected"])
        if ok:
            correct += 1
        else:
            wrong.append(
                {
                    "q": qa["prompt"][:45],
                    "expected": qa["expected"],
                    "got": res["response"].strip()[:50],
                }
            )

        total_tps += res["tps"]
        total_ttft += res["ttft"]
        status = "✓" if ok else "✗"
        print(
            f"  {i+1:<4} {res['ttft']*1000:<11.0f} {res['tps']:<9.1f} {status:<5} "
            f"{res['response'].strip()[:50]}"
        )

    n = len(QA_PAIRS)
    avg_tps = total_tps / n
    avg_ttft = total_ttft / n
    accuracy = correct / n * 100

    if wrong:
        print(f"\n  Wrong answers ({len(wrong)}):")
        for w in wrong:
            print(f"    [✗] {w['q']}")
            print(f"        expected: {w['expected']}  got: {w['got']}")

    del llm

    return {
        "name": name,
        "load_s": load_s,
        "avg_ttft_ms": avg_ttft * 1000,
        "avg_tps": avg_tps,
        "correct": correct,
        "total": n,
        "accuracy": accuracy,
    }


def print_comparison(results: list[dict]) -> None:
    if not results:
        print("\nNo results to compare.")
        return

    ranked = sorted(results, key=lambda r: (r["accuracy"], r["avg_tps"]), reverse=True)
    col_name = max(max(len(r["name"]) for r in ranked), 10)

    header = (
        f"  {'Model':<{col_name}}  {'Load(s)':<9} {'TTFT(ms)':<10} "
        f"{'TPS':<8} {'Accuracy':<10} {'Score'}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print(f"\n\n{'#'*75}")
    print("  COMPARISON SUMMARY")
    print(f"{'#'*75}")
    print(header)
    print(sep)

    for rank, r in enumerate(ranked, 1):
        score = f"{r['correct']}/{r['total']}"
        print(
            f"  {rank}. {r['name']:<{col_name-3}}  "
            f"{r['load_s']:<9.1f} {r['avg_ttft_ms']:<10.0f} "
            f"{r['avg_tps']:<8.1f} {r['accuracy']:<10.0f}% {score}"
        )

    print(sep)
    best_acc = ranked[0]
    best_tps = max(results, key=lambda r: r["avg_tps"])
    best_load = min(results, key=lambda r: r["load_s"])
    print(f"\n  Highest accuracy : {best_acc['name']}  ({best_acc['accuracy']:.0f}%)")
    print(f"  Fastest TPS      : {best_tps['name']}  ({best_tps['avg_tps']:.1f} tok/s)")
    print(f"  Fastest load     : {best_load['name']}  ({best_load['load_s']:.1f}s)")
    print(f"{'#'*75}\n")


# ===========================================================================
# ── SECTION 2: Generation quality metrics ─────────────────────────────────
# ===========================================================================


# ── 2.1  ROUGE-L ────────────────────────────────────────────────────────────


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence length (O(n*m) dynamic programming)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Space-optimised: only two rows needed
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


def rouge_l(hypothesis: str, reference: str) -> dict[str, float]:
    """
    ROUGE-L: Longest Common Subsequence F1 between hypothesis and reference.

    Rationale: captures in-order word overlap without requiring contiguous
    n-grams, making it robust to syntactic variation common in PT-BR
    paraphrases. Standard metric for summarisation and open-ended QA.

    Returns precision, recall, and F1 (beta=1).
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ── 2.2  BERTScore (approximation without heavy dependencies) ───────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tf_idf_vector(
    text: str, vocab: dict[str, int], idf: dict[str, float]
) -> list[float]:
    """TF-IDF vector over a shared vocabulary."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    vec = [0.0] * len(vocab)
    for t, freq in tf.items():
        if t in vocab:
            vec[vocab[t]] = (freq / total) * idf.get(t, 1.0)
    return vec


def bert_score_approx(
    hypothesis: str,
    reference: str,
    embed_fn=None,
) -> float:
    """
    BERTScore F1 approximation.

    When an embed_fn is supplied (signature: (list[str]) -> list[list[float]]),
    uses real sentence embeddings (e.g. from the project's EmbeddingModel).
    Falls back to TF-IDF cosine similarity when no embed_fn is provided.

    Rationale: BERTScore (Zhang et al., 2020) correlates better with human
    judgement than ROUGE for open-ended answers in non-English languages,
    because it captures semantic equivalence rather than exact word overlap.

    Reference: https://arxiv.org/abs/1904.09675
    """
    if embed_fn is not None:
        try:
            vecs = embed_fn([hypothesis, reference])
            return float(_cosine_similarity(vecs[0], vecs[1]))
        except Exception:
            pass  # fall through to TF-IDF approximation

    # TF-IDF fallback: build vocabulary from both texts
    all_tokens = re.findall(r"\b\w+\b", (hypothesis + " " + reference).lower())
    vocab = {t: i for i, t in enumerate(sorted(set(all_tokens)))}

    # IDF: log(2 / (1 + df)) — with 2 documents
    hyp_set = set(re.findall(r"\b\w+\b", hypothesis.lower()))
    ref_set = set(re.findall(r"\b\w+\b", reference.lower()))
    idf: dict[str, float] = {}
    for t in vocab:
        df = (t in hyp_set) + (t in ref_set)
        idf[t] = math.log((2 + 1) / (df + 1)) + 1  # smoothed

    vec_h = _tf_idf_vector(hypothesis, vocab, idf)
    vec_r = _tf_idf_vector(reference, vocab, idf)
    return float(_cosine_similarity(vec_h, vec_r))


# ── 2.3  Faithfulness ───────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for PT-BR text."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _token_overlap(a: str, b: str, min_len: int = 4) -> float:
    """Jaccard token overlap between two strings, ignoring short tokens."""
    ta = {t for t in re.findall(r"\b\w+\b", a.lower()) if len(t) >= min_len}
    tb = {t for t in re.findall(r"\b\w+\b", b.lower()) if len(t) >= min_len}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def faithfulness(response: str, context: str, threshold: float = 0.08) -> float:
    """
    Faithfulness score: fraction of response sentences that have sufficient
    lexical overlap with the retrieved context.

    Rationale: a RAG system should ground its answers in the provided context.
    This metric directly operationalises hallucination detection without
    requiring a second LLM call (unlike LLM-as-judge approaches).

    Inspired by: RAGAS (Es et al., 2023) — https://arxiv.org/abs/2309.15217

    Args:
        response:  The model's generated answer.
        context:   The concatenated retrieved chunks.
        threshold: Minimum Jaccard overlap for a sentence to be considered
                   "supported". Default 0.08 is intentionally tolerant for
                   long textbook chunks, where Jaccard is diluted by many
                   unrelated context tokens.

    Returns:
        Float in [0, 1]. 1.0 = fully grounded; 0.0 = no overlap.
    """
    sentences = _split_sentences(response)
    if not sentences:
        return 0.0
    context_tokens = {t for t in re.findall(r"\b\w+\b", context.lower()) if len(t) >= 4}
    supported = 0
    for s in sentences:
        sentence_tokens = {t for t in re.findall(r"\b\w+\b", s.lower()) if len(t) >= 4}
        if not sentence_tokens:
            continue
        overlap = _token_overlap(s, context)
        coverage = len(sentence_tokens & context_tokens) / len(sentence_tokens)
        if overlap >= threshold or coverage >= 0.35:
            supported += 1
    return supported / len(sentences)


# ── 2.4  Answer Relevance ───────────────────────────────────────────────────


def answer_relevance(
    response: str,
    query: str,
    embed_fn=None,
) -> float:
    """
    Answer Relevance: cosine similarity between response and query embeddings.

    Rationale: measures whether the response is topically aligned with the
    question, regardless of factual correctness. A response can be faithful
    to the context but still off-topic. Uses the same embedding model as
    the RAG pipeline when available, ensuring consistency.

    Inspired by: RAGAS (Es et al., 2023).

    Returns float in [-1, 1]; higher = more relevant.
    """
    return bert_score_approx(response, query, embed_fn=embed_fn)


# ── 2.5  Exact Match ────────────────────────────────────────────────────────


def exact_match(response: str, expected: list[str]) -> bool:
    """
    Exact Match: checks whether any expected answer string appears in the
    normalised response.

    Normalisation handles subscripts (₂ → 2), case, and accents.
    Used for factual questions with unambiguous answers (dates, names,
    chemical formulas). Not appropriate for open-ended explanations.

    Returns True if at least one expected string is found in the response.
    """
    if not expected:
        return True  # no ground truth → not penalised
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    norm = response.translate(subscript_map).lower()
    return any(e.lower() in norm for e in expected)


# ── 2.6  No-Context Rate ────────────────────────────────────────────────────


def is_no_context_response(response: str) -> bool:
    """
    Returns True when the model correctly refused to answer because the
    context did not contain the information.

    Rationale: the guardrail (_NO_CONTEXT_RESPONSE in pipeline.py) is a
    critical safety feature. We track its precision (did it fire on
    no-context items?) and recall (did it suppress it on valid items?).
    """
    return _NO_CONTEXT_MARKER in response.lower()


# ── 2.7  Peak RSS memory ────────────────────────────────────────────────────


def peak_rss_mb() -> float:
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1_048_576  # bytes → MB
        return usage.ru_maxrss / 1_024  # kB → MB

    # Fallback para Windows
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ── 2.8  Context relevance ────────────────────────────────────────────────────


def context_relevance(
    context: str,
    query: str,
    embed_fn=None,
) -> float:
    """
    Context Relevance: mede o quanto o contexto recuperado é relevante
    para a pergunta.

    Implementação:
    - Usa embeddings (se disponíveis)
    - Fallback para TF-IDF (igual ao bert_score_approx)

    Rationale:
    Um bom sistema RAG deve recuperar contexto altamente relevante
    antes de gerar a resposta.

    Inspired by: RAGAS (Es et al., 2023)

    Returns float em [-1, 1]
    """

    return bert_score_approx(context, query, embed_fn=embed_fn)


# ===========================================================================
# ── SECTION 3: Generation benchmark runner ─────────────────────────────────
# ===========================================================================


def _generate_with_context(llm, query: str, context: str) -> dict:
    """Run a single RAG-style generation and measure latency + memory."""
    system_prompt = _SYSTEM_TEMPLATE.format(context=context)
    messages = [{"role": "user", "content": f"{system_prompt}\n\nPergunta: {query}"}]

    ttft = None
    response_text = ""
    token_count = 0
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()

    try:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,  # matches pipeline default for factual RAG
            top_p=0.95,
            repeat_penalty=1.1,
            stream=True,
        )
        for chunk in stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            token = choices[0].get("delta", {}).get("content", "")
            if token:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                response_text += token
                token_count += 1
    except Exception as e:
        response_text = f"[ERROR: {e}]"

    total_s = time.perf_counter() - t0
    rss_after = peak_rss_mb()

    return {
        "response": response_text,
        "ttft_ms": (ttft or 0.0) * 1000,
        "tps": token_count / total_s if total_s > 0 else 0,
        "total_s": total_s,
        "tokens": token_count,
        "peak_rss_mb": max(rss_before, rss_after),
    }


def benchmark_generation(
    model_path: Path,
    n_ctx: int,
    threads: int,
    gpu_layers: int,
    dataset: list[dict],
    embed_fn=None,
) -> dict | None:
    """
    Run the generation quality benchmark for a single model.

    Computes per-item and aggregate metrics:
      - ROUGE-L F1
      - BERTScore-F1 (TF-IDF approximation or real embeddings)
      - Faithfulness
      - Answer Relevance
      - Exact Match
      - No-Context Precision (guardrail fires correctly on no-context items)
      - No-Context Recall (guardrail does NOT fire on answerable items)
      - TTFT (ms), TPS, Peak RSS (MB)
    """
    from llama_cpp import Llama

    name = model_path.name
    print(f"\n{'='*75}")
    print(f"  Generation Benchmark: {name}")
    print(f"{'='*75}")

    try:
        rss_before_load = peak_rss_mb()
        t_load = time.perf_counter()
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        load_s = time.perf_counter() - t_load
        rss_after_load = peak_rss_mb()
        print(f"  Loaded in {load_s:.1f}s | RSS after load: {rss_after_load:.0f} MB")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    # Warmup
    try:
        _generate_with_context(llm, "teste", "contexto de teste")
    except Exception:
        pass

    results_per_item = []
    col = f"  {'#':<3} {'Cat':<12} {'ROUGE-L':<9} {'BERT':<9} {'Faith':<8} {'Rel':<8} {'EM':<5} {'TTFT':<8} {'TPS':<7} {'Rel':<8} {'CtxRel':<8} RESPONSE"
    print(f"\n{col}")
    print(f"  {'-'*110}")

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        context = item["context"]
        reference = item.get("reference")
        expected = item.get("expected", [])
        category = item.get("category", "other")

        sys.stdout.write(f"  [{i}/{len(dataset)}] Generating...\r")
        sys.stdout.flush()

        gen = _generate_with_context(llm, query, context)
        response = gen["response"]

        # ── compute metrics ────────────────────────────────────────────
        rl = rouge_l(response, reference) if reference else {"f1": None}
        bs = (
            bert_score_approx(response, reference, embed_fn=embed_fn)
            if reference
            else None
        )
        faith = faithfulness(response, context)
        rel = answer_relevance(response, query, embed_fn=embed_fn)
        em = exact_match(response, expected)
        no_ctx = is_no_context_response(response)
        ctx_rel = context_relevance(context, query, embed_fn=embed_fn)

        item_result = {
            "query": query,
            "category": category,
            "response": response,
            "rouge_l_f1": rl["f1"],
            "bert_score": bs,
            "faithfulness": faith,
            "answer_relevance": rel,
            "exact_match": em,
            "no_context_response": no_ctx,
            "ttft_ms": gen["ttft_ms"],
            "tps": gen["tps"],
            "peak_rss_mb": gen["peak_rss_mb"],
            "context_relevance": ctx_rel,
        }
        results_per_item.append(item_result)

        def _fmt(v, pct=True):
            if v is None:
                return "  N/A  "
            return f"{v*100:5.1f}%" if pct else f"{v:6.2f}"

        em_sym = "✓" if em else "✗"
        if not expected:
            em_sym = "–"

        print(
            f"  {i:<3} {category:<12} "
            f"{_fmt(rl['f1'])} {_fmt(bs)} {_fmt(faith)} {_fmt(rel)} "
            f"{em_sym:<5} {gen['ttft_ms']:6.0f}ms {gen['tps']:5.1f}  "
            f"{response[:40].replace(chr(10),' ')!r}"
            f"{_fmt(rel)} {_fmt(ctx_rel)} "
        )

    del llm

    # ── Aggregate ─────────────────────────────────────────────────────
    answerable = [r for r in results_per_item if r["category"] != "no_context"]
    no_ctx_items = [r for r in results_per_item if r["category"] == "no_context"]

    def _mean(lst):
        lst = [x for x in lst if x is not None]
        return sum(lst) / len(lst) if lst else None

    agg = {
        "model": name,
        "n_items": len(results_per_item),
        # Quality (answerable items only)
        "rouge_l_f1": _mean([r["rouge_l_f1"] for r in answerable]),
        "bert_score_f1": _mean([r["bert_score"] for r in answerable]),
        "faithfulness": _mean([r["faithfulness"] for r in results_per_item]),
        "answer_relevance": _mean([r["answer_relevance"] for r in answerable]),
        "exact_match_rate": (
            sum(r["exact_match"] for r in answerable) / len(answerable)
            if answerable
            else None
        ),
        # Guardrail
        "no_ctx_precision": (
            sum(r["no_context_response"] for r in no_ctx_items) / len(no_ctx_items)
            if no_ctx_items
            else None
        ),
        "no_ctx_recall": (
            1.0 - sum(r["no_context_response"] for r in answerable) / len(answerable)
            if answerable
            else None
        ),
        # Efficiency
        "avg_ttft_ms": _mean([r["ttft_ms"] for r in results_per_item]),
        "avg_tps": _mean([r["tps"] for r in results_per_item]),
        "peak_rss_mb": max(r["peak_rss_mb"] for r in results_per_item),
        "context_relevance": _mean([r["context_relevance"] for r in results_per_item]),
        # Per-item detail
        "per_item": results_per_item,
    }
    return agg


def benchmark_real_rag(
    model_path: Path,
    n_ctx: int,
    threads: int,
    gpu_layers: int,
    dataset: list[dict],
    settings,
    embed_fn=None,
) -> dict | None:
    """Run the same generation metrics through the real project RAG pipeline.

    Unlike benchmark_generation(), this does not inject item["context"] directly.
    It loads the project VectorStore/Retriever, lets the pipeline retrieve chunks,
    builds the real prompt, applies history/guardrails, and then scores the answer
    against the actually retrieved context.
    """
    from core.conversation import ConversationManager
    from llm.local_model import LocalModel
    from rag.embeddings import EmbeddingModel
    from rag.pipeline import RAGPipeline
    from rag.retriever import Retriever
    from rag.vectorstore import VectorStore

    class TimedLocalModel:
        """Wrap LocalModel so the real pipeline still reports TTFT/TPS."""

        def __init__(self, inner: LocalModel) -> None:
            self._inner = inner
            self.reset_metrics()

        def reset_metrics(self) -> None:
            self.last_metrics = {
                "ttft_ms": 0.0,
                "tps": 0.0,
                "total_s": 0.0,
                "tokens": 0,
                "peak_rss_mb": peak_rss_mb(),
            }

        def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> str:
            self.reset_metrics()
            ttft = None
            response_text = ""
            token_count = 0
            rss_before = peak_rss_mb()
            t0 = time.perf_counter()

            token_stream = self._inner.chat(messages, stream=True, **kwargs)
            for token in token_stream:
                if token:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    response_text += token
                    token_count += 1

            total_s = time.perf_counter() - t0
            rss_after = peak_rss_mb()
            self.last_metrics = {
                "ttft_ms": (ttft or 0.0) * 1000,
                "tps": token_count / total_s if total_s > 0 else 0.0,
                "total_s": total_s,
                "tokens": token_count,
                "peak_rss_mb": max(rss_before, rss_after),
            }
            return response_text

    name = f"{model_path.name} [real-rag]"
    print(f"\n{'='*75}")
    print(f"  Real RAG Benchmark: {model_path.name}")
    print(f"{'='*75}")

    try:
        em = EmbeddingModel(
            model_name=settings.embed_model_name,
            cache_dir=settings.embed_cache_dir,
            batch_size=settings.embed_batch_size,
            use_disk_cache=settings.embed_disk_cache,
        )
        vs = VectorStore(index_dir=settings.index_dir, embedding_dim=em.dimension)
        retriever = Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=settings.top_k,
            candidate_multiplier=settings.candidate_multiplier,
            min_score=settings.min_score,
            lexical_weight=settings.lexical_weight,
        )
        t_load = time.perf_counter()
        base_llm = LocalModel(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
        )
        load_s = time.perf_counter() - t_load
        rss_after_load = peak_rss_mb()
        llm = TimedLocalModel(base_llm)
        print(f"  Loaded in {load_s:.1f}s | RSS after load: {rss_after_load:.0f} MB")
    except Exception as e:
        print(f"  FAILED to load RAG components: {e}")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    Path(tmp.name).write_text("[]", encoding="utf-8")
    conv_manager = ConversationManager(storage_path=tmp.name)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        conv_manager=conv_manager,
        max_context_chars=settings.max_context_chars,
        max_history_turns=settings.max_history_turns,
    )

    results_per_item = []
    col = (
        f"  {'#':<3} {'Cat':<12} {'ROUGE-L':<9} {'BERT':<9} "
        f"{'Faith':<8} {'Rel':<8} {'EM':<5} {'TTFT':<8} "
        f"{'TPS':<7} {'CtxRel':<8} RESPONSE"
    )
    print(f"\n{col}")
    print(f"  {'-'*125}")

    try:
        for i, item in enumerate(dataset, 1):
            query = item["query"]
            reference = item.get("reference")
            expected = item.get("expected", [])
            category = item.get("category", "other")

            chunks = retriever.retrieve(query)
            context = "\n\n".join(c.get("text", "") for c in chunks)
            llm.reset_metrics()
            response = pipeline.chat(f"speed-real-rag-{i}", query)
            gen = llm.last_metrics

            rl = rouge_l(response, reference) if reference else {"f1": None}
            bs = (
                bert_score_approx(response, reference, embed_fn=embed_fn)
                if reference
                else None
            )
            faith = faithfulness(response, context) if context else 0.0
            rel = answer_relevance(response, query, embed_fn=embed_fn)
            em = exact_match(response, expected)
            no_ctx = is_no_context_response(response)
            ctx_rel = (
                context_relevance(context, query, embed_fn=embed_fn) if context else 0.0
            )

            results_per_item.append(
                {
                    "query": query,
                    "category": category,
                    "response": response,
                    "retrieved_context": context,
                    "retrieved_scores": [c.get("score", 0.0) for c in chunks],
                    "rouge_l_f1": rl["f1"],
                    "bert_score": bs,
                    "faithfulness": faith,
                    "answer_relevance": rel,
                    "exact_match": em,
                    "no_context_response": no_ctx,
                    "ttft_ms": gen["ttft_ms"],
                    "tps": gen["tps"],
                    "peak_rss_mb": gen["peak_rss_mb"],
                    "context_relevance": ctx_rel,
                }
            )

            def _fmt(v):
                return "  N/A  " if v is None else f"{v*100:5.1f}%"

            em_sym = "✓" if em else "✗"
            if not expected:
                em_sym = "–"
            print(
                f"  {i:<3} {category:<12} "
                f"{_fmt(rl['f1'])} {_fmt(bs)} {_fmt(faith)} {_fmt(rel)} "
                f"{em_sym:<5} {gen['ttft_ms']:6.0f}ms {gen['tps']:5.1f}  "
                f"{_fmt(ctx_rel)} {response[:60].replace(chr(10), ' ')!r}"
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    answerable = [r for r in results_per_item if r["category"] != "no_context"]
    no_ctx_items = [r for r in results_per_item if r["category"] == "no_context"]

    def _mean(lst):
        lst = [x for x in lst if x is not None]
        return sum(lst) / len(lst) if lst else None

    return {
        "model": name,
        "n_items": len(results_per_item),
        "rouge_l_f1": _mean([r["rouge_l_f1"] for r in answerable]),
        "bert_score_f1": _mean([r["bert_score"] for r in answerable]),
        "faithfulness": _mean([r["faithfulness"] for r in results_per_item]),
        "answer_relevance": _mean([r["answer_relevance"] for r in answerable]),
        "exact_match_rate": (
            sum(r["exact_match"] for r in answerable) / len(answerable)
            if answerable
            else None
        ),
        "no_ctx_precision": (
            sum(r["no_context_response"] for r in no_ctx_items) / len(no_ctx_items)
            if no_ctx_items
            else None
        ),
        "no_ctx_recall": (
            1.0 - sum(r["no_context_response"] for r in answerable) / len(answerable)
            if answerable
            else None
        ),
        "avg_ttft_ms": _mean([r["ttft_ms"] for r in results_per_item]),
        "avg_tps": _mean([r["tps"] for r in results_per_item]),
        "peak_rss_mb": max(r["peak_rss_mb"] for r in results_per_item),
        "context_relevance": _mean([r["context_relevance"] for r in results_per_item]),
        "per_item": results_per_item,
    }


def print_generation_report(results: list[dict]) -> None:
    """Print aggregate generation benchmark table, with per-metric explanation."""
    sep = "═" * 80
    thin = "─" * 80

    print(f"\n\n{sep}")
    print("  GENERATION QUALITY BENCHMARK — AGGREGATE RESULTS")
    print(sep)

    metrics_meta = [
        (
            "rouge_l_f1",
            "ROUGE-L F1",
            True,
            "LCS overlap with reference. Higher = better fluency/coverage.",
        ),
        (
            "bert_score_f1",
            "BERTScore-F1",
            True,
            "Semantic similarity (TF-IDF approx). Higher = closer meaning.",
        ),
        (
            "faithfulness",
            "Faithfulness",
            True,
            "Fraction of response sentences grounded in context. Measures hallucination.",
        ),
        (
            "answer_relevance",
            "Answer Relevance",
            True,
            "Semantic sim(response, query). Higher = more topically on-target.",
        ),
        (
            "exact_match_rate",
            "Exact Match Rate",
            True,
            "Fraction of factual items where expected entity appears in response.",
        ),
        (
            "no_ctx_precision",
            "No-Ctx Precision",
            True,
            "Guardrail fired correctly on no-context items (should be ~1.0).",
        ),
        (
            "no_ctx_recall",
            "No-Ctx Recall",
            True,
            "Guardrail did NOT fire on answerable items (should be ~1.0).",
        ),
        (
            "avg_ttft_ms",
            "Avg TTFT (ms)",
            False,
            "Time to first token. Critical for interactive UX on RPi.",
        ),
        (
            "avg_tps",
            "Avg TPS",
            False,
            "Tokens per second. Higher = faster full response.",
        ),
        (
            "peak_rss_mb",
            "Peak RSS (MB)",
            False,
            "Peak resident memory. Keep below ~3 GB on Raspberry Pi 5.",
        ),
        (
            "context_relevance",
            "Context Relevance",
            True,
            "Semantic sim(context, query). Mede qualidade da recuperação.",
        ),
    ]

    # Header row
    name_col = max(len(r["model"]) for r in results)
    name_col = max(name_col, 10)
    row_h = f"  {'Metric':<25}" + "".join(
        f"  {r['model'][:name_col]:>{name_col}}" for r in results
    )
    print(row_h)
    print(thin)

    for key, label, higher_better, explanation in metrics_meta:
        vals = [r.get(key) for r in results]
        row = f"  {label:<25}"
        valid_vals = [v for v in vals if v is not None]
        best = (
            max(valid_vals)
            if higher_better and valid_vals
            else (min(valid_vals) if valid_vals else None)
        )
        for v in vals:
            if v is None:
                cell = "    N/A  "
            elif key in ("avg_ttft_ms", "peak_rss_mb"):
                cell = f"{v:>9.1f}"
            elif key == "avg_tps":
                cell = f"{v:>9.1f}"
            else:
                mark = " ★" if (v == best and len(results) > 1) else "  "
                cell = f"{v*100:>7.1f}%{mark}"
            row += f"  {cell:>{name_col}}"
        print(row)
        print(f"  {'':25}  {explanation}")
        print()

    print(sep)

    # Category breakdown
    print("\n  FAITHFULNESS BY CATEGORY")
    print(thin)
    cats = sorted({item["category"] for r in results for item in r["per_item"]})
    for cat in cats:
        row = f"  {cat:<20}"
        for r in results:
            items = [i for i in r["per_item"] if i["category"] == cat]
            if not items:
                row += f"  {'N/A':>{name_col}}"
            else:
                avg = sum(i["faithfulness"] for i in items) / len(items)
                row += f"  {avg*100:>{name_col-1}.1f}%"
        print(row)
    print(sep)


def compare_generation_results(results: list[dict]) -> None:
    """Print relative improvement between first and subsequent models."""
    if len(results) < 2:
        return
    baseline = results[0]
    print(f"\n  RELATIVE IMPROVEMENT vs. {baseline['model']}")
    print("─" * 60)
    for r in results[1:]:
        print(f"  vs. {r['model']}")
        for key, label in [
            ("rouge_l_f1", "ROUGE-L F1"),
            ("bert_score_f1", "BERTScore-F1"),
            ("faithfulness", "Faithfulness"),
            ("avg_tps", "TPS"),
        ]:
            b = baseline.get(key)
            c = r.get(key)
            if b is not None and c is not None and b != 0:
                delta = (c - b) / abs(b) * 100
                sym = "▲" if delta > 0 else "▼"
                print(f"    {label:<20}: {sym} {abs(delta):.1f}%")
    print()


# ===========================================================================
# ── SECTION 4: CLI ──────────────────────────────────────────────────────────
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GGUF models: latency/accuracy (default) "
            "and/or generation quality (--benchmark-generation)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory with .gguf files (default: models/)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to a specific .gguf file (overrides --models-dir for generation benchmark).",
    )
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--gpu-layers", type=int, default=None)

    # Generation benchmark flags
    gen_group = parser.add_argument_group("generation benchmark")
    gen_group.add_argument(
        "--benchmark-generation",
        action="store_true",
        help="Run generation quality benchmark (ROUGE-L, BERTScore, Faithfulness, …).",
    )
    gen_group.add_argument(
        "--benchmark-real-rag",
        action="store_true",
        help=(
            "Run generation metrics through the real RAG pipeline "
            "(retriever + prompt + guardrails), not simulated context."
        ),
    )
    gen_group.add_argument(
        "--all",
        action="store_true",
        help="Run latency/accuracy and the simulated generation benchmark.",
    )
    gen_group.add_argument(
        "--gen-dataset",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file with generation benchmark items. "
            "Each item must have: query, context, reference (nullable), "
            "expected (list[str]), category (str). "
            "Defaults to GENERATION_DATASET for --benchmark-generation and "
            "REAL_RAG_DATASET for --benchmark-real-rag."
        ),
    )
    gen_group.add_argument(
        "--save-gen-report",
        default=None,
        metavar="PATH",
        help="Save generation benchmark report to this JSON file.",
    )
    gen_group.add_argument(
        "--use-project-embeddings",
        action="store_true",
        help=(
            "Use the project's EmbeddingModel for BERTScore and Answer Relevance "
            "(requires EMBED_CACHE_DIR to be configured in .env). "
            "Without this flag, a TF-IDF approximation is used instead."
        ),
    )
    return parser


def main() -> None:
    settings = get_settings()
    setup_logging("WARNING")

    parser = _build_parser()
    args = parser.parse_args()

    n_ctx = args.n_ctx or settings.n_ctx
    threads = args.threads or settings.n_threads
    gpu_layers = args.gpu_layers or settings.n_gpu_layers

    base_dir = Path(__file__).parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else base_dir / "models"

    # ── Resolve GGUF files ────────────────────────────────────────────────
    if args.model:
        gguf_files = [Path(args.model)]
        if not gguf_files[0].exists():
            print(f"ERROR: model file not found: {args.model}")
            sys.exit(1)
    else:
        gguf_files = sorted(models_dir.glob("*.gguf"))
        if not gguf_files:
            print(f"No .gguf files found in '{models_dir}'")
            sys.exit(1)

    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        print(
            "ERROR: llama-cpp-python is not installed. Run: pip install llama-cpp-python"
        )
        sys.exit(1)

    run_latency = not (args.benchmark_generation or args.benchmark_real_rag) or args.all
    run_generation = args.benchmark_generation or args.all
    run_real_rag = args.benchmark_real_rag

    # ── Print header ───────────────────────────────────────────────────────
    print(f"\nFound {len(gguf_files)} model(s) in '{models_dir}':")
    for f in gguf_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name}  ({size_mb:.0f} MB)")
    print(f"\nSettings: n_ctx={n_ctx}  threads={threads}  gpu_layers={gpu_layers}")
    print(
        f"Modes: latency={'yes' if run_latency else 'no'}  "
        f"generation={'yes' if run_generation else 'no'}  "
        f"real_rag={'yes' if run_real_rag else 'no'}"
    )

    # ── Latency / accuracy benchmark (original) ────────────────────────────
    if run_latency:
        print(f"\nQ&A pairs: {len(QA_PAIRS)}")
        latency_results = []
        for model_path in gguf_files:
            result = benchmark_model(model_path, n_ctx, threads, gpu_layers)
            if result:
                latency_results.append(result)
        if latency_results:
            print_comparison(latency_results)

    generation_dataset = None
    real_rag_dataset = None
    embed_fn = None
    if run_generation or run_real_rag:
        if args.gen_dataset:
            try:
                with open(args.gen_dataset, encoding="utf-8") as f:
                    loaded_dataset = json.load(f)
                generation_dataset = loaded_dataset
                real_rag_dataset = loaded_dataset
                print(
                    f"\nGeneration dataset: {args.gen_dataset} "
                    f"({len(loaded_dataset)} items)"
                )
            except Exception as e:
                print(f"ERROR loading dataset: {e}")
                sys.exit(1)
        else:
            if run_generation:
                generation_dataset = GENERATION_DATASET
                print(
                    f"\nGeneration dataset: simulated built-in "
                    f"({len(generation_dataset)} items)"
                )
            if run_real_rag:
                real_rag_dataset = REAL_RAG_DATASET
                print(
                    f"\nReal RAG dataset: corpus built-in "
                    f"({len(real_rag_dataset)} items)"
                )

        if args.use_project_embeddings:
            try:
                from rag.embeddings import EmbeddingModel

                em = EmbeddingModel(
                    model_name=settings.embed_model_name,
                    cache_dir=settings.embed_cache_dir,
                    batch_size=settings.embed_batch_size,
                    use_disk_cache=settings.embed_disk_cache,
                )
                embed_fn = em.embed
                print(f"  Using project embeddings: {settings.embed_model_name}")
            except Exception as e:
                print(
                    f"  WARNING: could not load project embeddings ({e}). Using TF-IDF fallback."
                )

    all_report_results = []

    # ── Generation quality benchmark ───────────────────────────────────────
    if run_generation:
        gen_results = []
        for model_path in gguf_files:
            result = benchmark_generation(
                model_path,
                n_ctx,
                threads,
                gpu_layers,
                generation_dataset,
                embed_fn=embed_fn,
            )
            if result:
                gen_results.append(result)

        if gen_results:
            print_generation_report(gen_results)
            if len(gen_results) > 1:
                compare_generation_results(gen_results)
            all_report_results.extend(gen_results)

    # ── Real RAG benchmark ─────────────────────────────────────────────────
    if run_real_rag:
        real_rag_results = []
        for model_path in gguf_files:
            result = benchmark_real_rag(
                model_path,
                n_ctx,
                threads,
                gpu_layers,
                real_rag_dataset,
                settings,
                embed_fn=embed_fn,
            )
            if result:
                real_rag_results.append(result)

        if real_rag_results:
            print_generation_report(real_rag_results)
            if len(real_rag_results) > 1:
                compare_generation_results(real_rag_results)
            all_report_results.extend(real_rag_results)

    # Save report
    if args.save_gen_report and all_report_results:
        out = Path(args.save_gen_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Remove per_item detail to keep file manageable; include only aggregates
        serialisable = [
            {k: v for k, v in r.items() if k != "per_item"} for r in all_report_results
        ]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        print(f"\n  Generation report saved to: {out}")


if __name__ == "__main__":
    main()
