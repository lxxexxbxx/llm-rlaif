"""평가 결과 JSON을 집계해 비교표와 그래프를 생성합니다.

GPU가 필요 없으므로 맥북 로컬에서 바로 실행됩니다.

    python scripts/analyze_results.py
"""
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CHARTS = RESULTS / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Base Gemma3-4B": RESULTS / "eval_base_4b.json",
    "+ PPO":          RESULTS / "eval_ppo.json",
}


def summarize(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    acc = [r["accuracy"] for r in rows if isinstance(r.get("accuracy"), (int, float))]
    con = [r["conciseness"] for r in rows if isinstance(r.get("conciseness"), (int, float))]
    return {
        "n": len(rows),
        "accuracy": round(mean(acc), 3),
        "conciseness": round(mean(con), 3),
        "acc_zero": sum(1 for a in acc if a == 0),
    }


def main() -> None:
    summary = {}
    for label, path in RUNS.items():
        if not path.exists():
            print(f"[skip] {path.name} 없음")
            continue
        summary[label] = summarize(path)

    header = f"{'구성':<18}{'n':>5}{'Accuracy':>11}{'Conciseness':>13}{'Acc=0':>8}"
    print(header)
    print("-" * len(header))
    for label, s in summary.items():
        print(f"{label:<18}{s['n']:>5}{s['accuracy']:>11}{s['conciseness']:>13}{s['acc_zero']:>8}")

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out.relative_to(REPO)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 미설치 — 그래프 생략")
        return

    labels = list(summary)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], [summary[l]["accuracy"] for l in labels], width=0.4, label="Accuracy")
    ax.bar([i + 0.2 for i in x], [summary[l]["conciseness"] for l in labels], width=0.4, label="Conciseness")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("score (0-10)"); ax.set_title("Base vs PPO (n=50)")
    ax.legend(); fig.tight_layout()
    path = CHARTS / "summary_regenerated.png"
    fig.savefig(path, dpi=150)
    print(f"저장: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
