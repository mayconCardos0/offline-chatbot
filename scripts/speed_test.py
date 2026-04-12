"""
Speed & accuracy benchmark — runs ALL GGUF models in models/ and compares results.

Usage (from offline-chatbot/):
    python scripts/speed_test.py [--n-ctx N] [--threads N] [--gpu-layers N] [--models-dir PATH]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging

# ---------------------------------------------------------------------------
# Q&A pairs
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {"prompt": "Quanto é 15 + 27? Responda apenas com o número.", "expected": ["42"]},
    {"prompt": "Quanto é 144 dividido por 12? Responda apenas com o número.", "expected": ["12"]},
    {"prompt": "Qual é a raiz quadrada de 81? Responda apenas com o número.", "expected": ["9"]},
    {"prompt": "Quantos números primos existem entre 1 e 10? Responda apenas com o número.", "expected": ["4"]},
    {"prompt": "Quanto é 7 x 8? Responda apenas com o número.", "expected": ["56"]},
    {"prompt": "Qual é a capital da França? Responda apenas com o nome da cidade.", "expected": ["Paris"]},
    {"prompt": "Qual é a capital do Japão? Responda apenas com o nome da cidade.", "expected": ["Toquio", "Tóquio"]},
    {"prompt": "Qual é o maior oceano da Terra? Responda apenas com o nome.", "expected": ["Pacifico", "Pacífico"]},
    {"prompt": "Em qual continente fica o Egito? Responda apenas com o nome do continente.", "expected": ["Africa"]},
    {"prompt": "Qual é o rio mais longo do mundo? Responda apenas com o nome.", "expected": ["Nilo", "Amazonas"]},
    {"prompt": "Qual é o símbolo químico do ouro? Responda apenas com o símbolo.", "expected": ["Au"]},
    {"prompt": "Qual é o símbolo químico da água? Responda apenas com a fórmula.", "expected": ["H2O"]},
    {"prompt": "Quantos planetas existem no nosso sistema solar? Responda apenas com o número.", "expected": ["8"]},
    {"prompt": "Qual é o ponto de ebulição da água em Celsius? Responda apenas com o número.", "expected": ["100"]},
    {"prompt": "Qual gás as plantas absorvem da atmosfera? Responda apenas com o nome.", "expected": ["carbon dioxide", "CO2"]},
    {"prompt": "Em que ano terminou a Segunda Guerra Mundial? Responda apenas com o ano.", "expected": ["1945"]},
    {"prompt": "Quem escreveu Romeu e Julieta? Responda apenas com o nome.", "expected": ["Shakespeare", "William Shakespeare"]},
    {"prompt": "Quantos lados tem um hexágono? Responda apenas com o número.", "expected": ["6"]},
    {"prompt": "Qual é o ponto de congelamento da água em Fahrenheit? Responda apenas com o número.", "expected": ["32"]},
    {"prompt": "Quantos dias há em um ano bissexto? Responda apenas com o número.", "expected": ["366"]},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def benchmark_model(model_path: Path, n_ctx: int, threads: int, gpu_layers: int) -> dict | None:
    """Load a model, run all QA pairs, return summary dict. Returns None on load failure."""
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

    # Warmup
    sys.stdout.write("  Warming up... ")
    sys.stdout.flush()
    try:
        run_single(llm, "hi")
        print("done")
    except Exception:
        print("skipped")

    # Run benchmark
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
            wrong.append({"q": qa["prompt"][:45], "expected": qa["expected"], "got": res["response"].strip()[:50]})

        total_tps += res["tps"]
        total_ttft += res["ttft"]
        status = "✓" if ok else "✗"
        print(f"  {i+1:<4} {res['ttft']*1000:<11.0f} {res['tps']:<9.1f} {status:<5} {res['response'].strip()[:50]}")

    n = len(QA_PAIRS)
    avg_tps   = total_tps / n
    avg_ttft  = total_ttft / n
    accuracy  = correct / n * 100

    if wrong:
        print(f"\n  Wrong answers ({len(wrong)}):")
        for w in wrong:
            print(f"    [✗] {w['q']}")
            print(f"        expected: {w['expected']}  got: {w['got']}")

    # Free memory
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

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    if not results:
        print("\nNo results to compare.")
        return

    # Sort by accuracy desc, then TPS desc
    ranked = sorted(results, key=lambda r: (r["accuracy"], r["avg_tps"]), reverse=True)

    col_name  = max(len(r["name"]) for r in ranked)
    col_name  = max(col_name, 10)

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    setup_logging("WARNING")

    parser = argparse.ArgumentParser(description="Benchmark all GGUF models and compare.")
    parser.add_argument("--models-dir", default=None, help="Directory containing .gguf files (default: models/)")
    parser.add_argument("--n-ctx",      type=int, default=settings.n_ctx)
    parser.add_argument("--threads",    type=int, default=settings.n_threads)
    parser.add_argument("--gpu-layers", type=int, default=settings.n_gpu_layers)
    args = parser.parse_args()

    base_dir   = Path(__file__).parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else base_dir / "models"

    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"No .gguf files found in '{models_dir}'")
        sys.exit(1)

    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        print("ERROR: llama-cpp-python is not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    print(f"\nFound {len(gguf_files)} model(s) in '{models_dir}':")
    for f in gguf_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name}  ({size_mb:.0f} MB)")

    print(f"\nBenchmark settings: n_ctx={args.n_ctx}  threads={args.threads}  gpu_layers={args.gpu_layers}")
    print(f"Q&A pairs: {len(QA_PAIRS)}")

    results = []
    for model_path in gguf_files:
        result = benchmark_model(model_path, args.n_ctx, args.threads, args.gpu_layers)
        if result:
            results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    main()
