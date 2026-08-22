"""
Speed & accuracy benchmark — runs ALL GGUF models in models/ and compares results.

Usage
─────
    python scripts/speed_test.py
    python scripts/speed_test.py --threads 4 --n-ctx 2048
    python scripts/speed_test.py --model models/gemma-2-2b-it-q4_k_m.gguf
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Q&A pairs (latency + accuracy benchmark)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF models: latency and Q&A accuracy.",
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
        help="Path to a specific .gguf file (overrides --models-dir).",
    )
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--gpu-layers", type=int, default=None)
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

    # ── Print header ───────────────────────────────────────────────────────
    print(f"\nFound {len(gguf_files)} model(s) in '{models_dir}':")
    for f in gguf_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name}  ({size_mb:.0f} MB)")
    print(f"\nSettings: n_ctx={n_ctx}  threads={threads}  gpu_layers={gpu_layers}")

    # ── Latency / accuracy benchmark ────────────────────────────────────────
    print(f"\nQ&A pairs: {len(QA_PAIRS)}")
    latency_results = []
    for model_path in gguf_files:
        result = benchmark_model(model_path, n_ctx, threads, gpu_layers)
        if result:
            latency_results.append(result)
    if latency_results:
        print_comparison(latency_results)


if __name__ == "__main__":
    main()
