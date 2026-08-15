"""Sample benchmark across all 5 difficulty tiers for README visualization.

Runs both recognizers (AI + own-code) on a per-tier sample of the held-out
test set and writes a results JSON suitable for charting.
"""
import json
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from dataset import test_set
from src.ai_recognizer import AIRecognizer
import src.owncode_recognizer as ocr
from src.metrics import levenshtein_similarity, symbol_accuracy

SAMPLE_PER_TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark_tiers.json"
SKIP_AI = "--skip-ai" in sys.argv

def main():
    tests = test_set("data")
    by_tier = defaultdict(list)
    for s in tests:
        by_tier[s["tier"]].append(s)
    tiers = ["clean", "noisy", "low_res", "black_bg", "white_bg"]

    rng = random.Random(42)
    sample = []
    for t in tiers:
        pool = by_tier[t]
        n = min(SAMPLE_PER_TIER, len(pool))
        sample.extend(rng.sample(pool, n))

    ai = AIRecognizer() if not SKIP_AI else None
    owncode = ocr.recognize
    # Warm up the own-code engine so its one-time template-bank build is not
    # charged to the first image, the same way the AI engine's model is already
    # resident in Ollama before timing starts.
    owncode(sample[0]["image_path"])

    results = {"tiers": tiers, "sample_per_tier": SAMPLE_PER_TIER,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "ai_available": not SKIP_AI,
               "per_tier": {}, "aggregate": {}}
    ai_agg = {"n": 0, "lev": [], "sym": [], "time": []}
    oc_agg = {"n": 0, "lev": [], "sym": [], "time": []}

    for t in tiers:
        samples = [s for s in sample if s["tier"] == t]
        if not samples:
            continue
        ai_rows = []
        oc_rows = []
        for s in samples:
            path = s["image_path"]
            gt = s["latex"]
            if ai:
                t0 = time.perf_counter()
                pred = ai.recognize(path)
                dt = time.perf_counter() - t0
                ai_rows.append((pred, gt, dt))
            t0 = time.perf_counter()
            pred = owncode(path)
            dt = time.perf_counter() - t0
            oc_rows.append((pred, gt, dt))

        tier_res = {}
        for name, rows in (("ai", ai_rows), ("owncode", oc_rows)):
            if not rows:
                continue
            lev = [levenshtein_similarity(p, g) for p, g, _ in rows]
            sym = [symbol_accuracy(p, g) for p, g, _ in rows]
            times = [t_ for _, _, t_ in rows]
            exact = sum(1 for p, g, _ in rows if p.strip() == g.strip())
            tier_res[name] = {
                "n": len(rows),
                "exact_match_rate": exact / len(rows),
                "mean_levenshtein_similarity": sum(lev) / len(lev),
                "mean_symbol_accuracy": sum(sym) / len(sym),
                "mean_seconds": sum(times) / len(times),
            }
            agg = ai_agg if name == "ai" else oc_agg
            agg["n"] += len(rows)
            agg["lev"].extend(lev); agg["sym"].extend(sym); agg["time"].extend(times)
        results["per_tier"][t] = tier_res

    for name, agg in (("ai", ai_agg), ("owncode", oc_agg)):
        if agg["n"]:
            results["aggregate"][name] = {
                "n": agg["n"],
                "mean_levenshtein_similarity": sum(agg["lev"]) / len(agg["lev"]),
                "mean_symbol_accuracy": sum(agg["sym"]) / len(agg["sym"]),
                "mean_seconds": sum(agg["time"]) / len(agg["time"]),
            }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["aggregate"], indent=2))
    print("saved", OUT)

if __name__ == "__main__":
    main()
