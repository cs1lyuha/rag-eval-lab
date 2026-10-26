"""Raportul Markdown din rezultatele grilei: efectul fiecărei dimensiuni și cele mai bune configurații."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable

COLUMNS = [("hit@k", "hit_at_k"), ("MRR", "mrr"), ("ctx prec.", "ctx_precision"), ("ctx recall", "ctx_recall"),
           ("answer recall", "answer_recall"), ("answer F1", "answer_f1"), ("refuz corect", "abstain_ok"),
           ("refuz greșit", "false_abstain"),
           ("tokeni ctx", "ctx_tokens"), ("$ / 1k întrebări", "usd_per_1k"), ("latență ms", "latency_ms")]


def _fmt(attr: str, value: float) -> str:
    if attr == "ctx_tokens":
        return f"{value:.0f}"
    if attr == "usd_per_1k":
        return f"{value:.3f}"
    if attr == "latency_ms":
        return f"{value:.2f}"
    return f"{value:.3f}"


def table(rows, label: Callable, header: str) -> list[str]:
    out = [f"| {header} | " + " | ".join(c for c, _ in COLUMNS) + " |",
           "|---|" + "---:|" * len(COLUMNS)]
    for r in rows:
        out.append(f"| {label(r)} | " + " | ".join(_fmt(a, getattr(r, a)) for _, a in COLUMNS) + " |")
    return out


def effect(rows, dim: str) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[getattr(r, dim)].append(r)
    out = [f"| {dim} | hit@k | MRR | ctx recall | answer recall | tokeni ctx |", "|---|---:|---:|---:|---:|---:|"]
    for key, rs in groups.items():
        m = lambda a: statistics.mean(getattr(r, a) for r in rs)  # noqa: E731
        out.append(f"| {key} | {m('hit_at_k'):.3f} | {m('mrr'):.3f} | {m('ctx_recall'):.3f} | "
                   f"{m('answer_recall'):.3f} | {m('ctx_tokens'):.0f} |")
    return out


def best(rows, n: int = 10):
    return sorted(rows, key=lambda r: (-r.answer_recall, -r.hit_at_k, r.ctx_tokens))[:n]


def pareto(rows):
    """Configurațiile pentru care nu există alta mai bună și mai ieftină în același timp."""
    front = []
    for r in rows:
        dominated = any(o.answer_recall >= r.answer_recall and o.usd_per_1k <= r.usd_per_1k and
                        (o.answer_recall > r.answer_recall or o.usd_per_1k < r.usd_per_1k) for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r.usd_per_1k)


def write_report(rows, path: Path, types=None, sweep=None, sweep_config: str = "") -> None:
    lines = ["# Rezultate grila RAG", "",
             f"{len(rows)} configurații × {len({r.chunking for r in rows})} strategii de chunking. "
             "Generator: extractiv (offline). Cost: estimat pentru Claude Sonnet 5.5 ($2 / $10 per 1M tokeni).", "",
             "## Top 10 după answer recall", ""]
    lines += table(best(rows), lambda r: r.key, "configurație")
    lines += ["", "## Frontiera Pareto (calitate vs. cost)", ""]
    lines += table(pareto(rows), lambda r: r.key, "configurație")
    for dim in ("chunking", "retriever", "rerank", "k"):
        lines += ["", f"## Efectul dimensiunii `{dim}` (medie pe restul grilei)", ""]
        lines += effect(rows, dim)
    if types:
        names = list(next(iter(types.values())))
        lines += ["", "## hit@3 pe tipuri de întrebări (chunking pe secțiuni, fără reranking)", "",
                  "| retriever | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for retriever, vals in types.items():
            lines.append(f"| {retriever} | " + " | ".join(f"{vals[n]:.3f}" for n in names) + " |")
    if sweep:
        lines += ["", f"## Pragul de refuz al generatorului (`{sweep_config}`)", "",
                  "| prag | refuz corect (fără răspuns) | refuz greșit (cu răspuns) | answer recall |",
                  "|---:|---:|---:|---:|"]
        for s_ in sweep:
            lines.append(f"| {s_['threshold']:.2f} | {s_['abstain_ok']:.2f} | {s_['false_abstain']:.3f} | "
                         f"{s_['answer_recall']:.3f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
