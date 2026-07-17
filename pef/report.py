"""Turn evaluation results into a Markdown summary or a JSON-serializable dict."""

from __future__ import annotations

from .criteria import CRITERIA
from .evaluator import PromptVersionResult


def to_json(results: list[PromptVersionResult]) -> dict:
    ranked = sorted(results, key=lambda r: r.composite, reverse=True)
    return {
        "criteria": list(CRITERIA),
        "results": [
            {
                "prompt_id": r.prompt_id,
                "composite": round(r.composite, 4),
                "mean_scores": {c: round(v, 4) for c, v in r.mean_scores.items()},
                "cases": [
                    {
                        "case_id": cr.case_id,
                        "primary_output": cr.primary_output,
                        "all_outputs": cr.all_outputs,
                        "scores": {c: round(v, 4) for c, v in cr.scores.items()},
                    }
                    for cr in r.case_results
                ],
            }
            for r in ranked
        ],
    }


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def to_markdown(results: list[PromptVersionResult], title: str = "Prompt Evaluation Results") -> str:
    ranked = sorted(results, key=lambda r: r.composite, reverse=True)

    lines = [f"# {title}", ""]

    header = ["Rank", "Prompt", "Composite"] + [c.replace("_", " ").title() for c in CRITERIA]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for rank, r in enumerate(ranked, start=1):
        row = [str(rank), r.prompt_id, _pct(r.composite)] + [_pct(r.mean_scores[c]) for c in CRITERIA]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-test-case detail")
    for r in ranked:
        lines.append("")
        lines.append(f"### {r.prompt_id}")
        for cr in r.case_results:
            lines.append("")
            lines.append(f"**Case `{cr.case_id}`**")
            lines.append("")
            lines.append(f"> {cr.primary_output}")
            lines.append("")
            score_line = ", ".join(f"{c.replace('_', ' ')}: {_pct(cr.scores[c])}" for c in CRITERIA)
            lines.append(f"_{score_line}_")

    return "\n".join(lines) + "\n"


def to_dataframe(results: list[PromptVersionResult]):
    """Optional pandas view of the mean scores. Requires `pandas` to be installed."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("to_dataframe() requires pandas. Install it with `pip install pandas`.") from exc

    ranked = sorted(results, key=lambda r: r.composite, reverse=True)
    rows = [{"prompt_id": r.prompt_id, "composite": r.composite, **r.mean_scores} for r in ranked]
    return pd.DataFrame(rows)
