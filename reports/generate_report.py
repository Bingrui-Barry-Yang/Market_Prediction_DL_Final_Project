"""GEPA run analysis report generator.

Reads a GEPA run directory and produces:
  - Individual PDF figures (matplotlib native PDF backend)
  - Combined single PDF report via PdfPages
  - LaTeX source (report.tex) that \includegraphics the figures
  - A small JSON summary of the computed metrics

The analyses are framed for prompt-optimization research:
  - Prompt length trajectory (words, chars)
  - Token-set Jaccard similarity vs. the seed
  - Structural feature counts per candidate (headers, bullets, code blocks,
    tables, numbered steps) to expose format evolution
  - Candidate x domain-keyword occurrence heatmap (lexical feature drift)
  - New-token bar chart: tokens present in best candidate but absent from seed
  - Iteration dynamics: subsample score before vs. after proposal, acceptance
  - Per-valset-task score heatmap across accepted candidates
  - Reflection-LM call dynamics: prompt/response length, cumulative chars

Usage:
  python reports/generate_report.py --run-dir outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# --- Analysis config --------------------------------------------------------

DEFAULT_KEYWORDS: list[str] = [
    "bearish",
    "bullish",
    "neutral",
    "forecast",
    "prediction",
    "forward-looking",
    "future",
    "analyst",
    "outlook",
    "target",
    "historical",
    "past",
    "retrospective",
    "already",
    "json",
    "integer",
    "score",
    "respond",
    "confidence",
    "strongly",
    "weakly",
    "article",
    "title",
    "text",
    "range",
    "bitcoin",
    "btc",
    "price",
    "example",
    "rule",
    "step",
]

STRUCTURAL_PATTERNS: dict[str, re.Pattern] = {
    "markdown_headers":   re.compile(r"^#{1,6}\s", re.MULTILINE),
    "bullet_items":       re.compile(r"^\s*[-*]\s", re.MULTILINE),
    "numbered_items":     re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    "code_fences":        re.compile(r"```"),
    "bold_spans":         re.compile(r"\*\*[^*]+\*\*"),
    "inline_code":        re.compile(r"`[^`\n]+`"),
    "table_pipes":        re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE),
}

# --- Data loading -----------------------------------------------------------


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read everything GEPA persisted under run_dir. Missing files return empty."""
    data: dict[str, Any] = {
        "candidates": [],
        "trace": [],
        "reflection": [],
        "per_task_outputs": {},
    }

    cand_path = run_dir / "candidates.json"
    if cand_path.exists():
        data["candidates"] = json.loads(cand_path.read_text(encoding="utf-8"))

    log_path = run_dir / "run_log.json"
    if log_path.exists():
        try:
            data["trace"] = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data["trace"] = []

    refl_path = run_dir / "reflection_transcripts.jsonl"
    if refl_path.exists():
        with refl_path.open(encoding="utf-8") as f:
            data["reflection"] = [json.loads(line) for line in f if line.strip()]

    task_root = run_dir / "generated_best_outputs_valset"
    if task_root.exists():
        for task_dir in sorted(task_root.glob("task_*")):
            entries = {}
            for entry in sorted(task_dir.glob("iter_*_prog_*.json")):
                m = re.match(r"iter_(\d+)_prog_(\d+)", entry.stem)
                if not m:
                    continue
                it, prog = int(m.group(1)), int(m.group(2))
                try:
                    payload = json.loads(entry.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                entries[(it, prog)] = payload
            data["per_task_outputs"][task_dir.name] = entries

    return data


def load_val_gold(data_path: Path) -> list[dict[str, Any]]:
    """Load the articles JSONL and return the val slice using GEPA's split rule.

    `scripts/run_gepa.py` sends every 5th row (0-indexed) to val, skipping rows
    without gold_score or gold_reasoning.
    """
    if not data_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    val: list[dict[str, Any]] = []
    kept = 0
    for i, row in enumerate(rows):
        gs = row.get("gold_score")
        gr = str(row.get("gold_reasoning", "")).strip()
        if gs is None or not gr:
            continue
        if kept % 5 == 0:
            val.append(row)
        kept += 1
    return val


# --- Metrics ----------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text.lower())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not (ta | tb):
        return 0.0
    return len(ta & tb) / len(ta | tb)


def structural_counts(text: str) -> dict[str, int]:
    return {name: len(pat.findall(text)) for name, pat in STRUCTURAL_PATTERNS.items()}


def keyword_counts(text: str, keywords: list[str]) -> list[int]:
    lower = text.lower()
    counts = []
    for kw in keywords:
        pat = re.compile(r"\b" + re.escape(kw) + r"\b")
        counts.append(len(pat.findall(lower)))
    return counts


def parse_score_response(response: str) -> int | None:
    """Mirrors SentimentScoreEvaluator's parse logic in scripts/run_gepa.py."""
    if not isinstance(response, str):
        return None
    raw = response.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
    try:
        obj = json.loads(raw)
        val = int(obj["score"])
        return val if 1 <= val <= 15 else None
    except Exception:
        return None


def score_against_gold(predicted: int | None, gold: int) -> float:
    if predicted is None:
        return 0.0
    if predicted == gold:
        return 1.0
    diff = abs(predicted - gold)
    return {1: 0.75, 2: 0.5, 3: 0.25}.get(diff, 0.0)


# --- Plots ------------------------------------------------------------------


_COMBINED_PDF: PdfPages | None = None


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    if _COMBINED_PDF is not None:
        _COMBINED_PDF.savefig(fig)
    plt.close(fig)


def plot_prompt_length(candidates: list[dict], out: Path) -> dict[str, Any]:
    words = [word_count(c["system_prompt"]) for c in candidates]
    chars = [len(c["system_prompt"]) for c in candidates]
    x = list(range(len(candidates)))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, words, "o-", color="C0", label="words")
    ax1.set_xlabel("Candidate index")
    ax1.set_ylabel("Word count", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(x, chars, "s--", color="C3", label="chars")
    ax2.set_ylabel("Char count", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Prompt length per candidate")
    ax1.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out)
    return {"words": words, "chars": chars}


def plot_jaccard_vs_seed(candidates: list[dict], out: Path) -> list[float]:
    seed = candidates[0]["system_prompt"] if candidates else ""
    vals = [jaccard(seed, c["system_prompt"]) for c in candidates]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(vals)), vals, "o-")
    ax.set_xlabel("Candidate index")
    ax.set_ylabel("Jaccard(tokens vs. seed)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Token overlap with seed prompt")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out)
    return vals


def plot_structural(candidates: list[dict], out: Path) -> dict[str, list[int]]:
    rows = [structural_counts(c["system_prompt"]) for c in candidates]
    names = list(STRUCTURAL_PATTERNS.keys())
    mat = np.array([[r[n] for n in names] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(candidates))
    width = 0.11
    for j, name in enumerate(names):
        ax.bar(x + (j - (len(names) - 1) / 2) * width, mat[:, j], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{i}" for i in range(len(candidates))])
    ax.set_ylabel("Count")
    ax.set_title("Structural features per candidate")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _save(fig, out)
    return {n: mat[:, j].tolist() for j, n in enumerate(names)}


def plot_keyword_heatmap(
    candidates: list[dict], keywords: list[str], out: Path
) -> np.ndarray:
    mat = np.array(
        [keyword_counts(c["system_prompt"], keywords) for c in candidates],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.35 * len(candidates) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(keywords)))
    ax.set_xticklabels(keywords, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels([f"c{i}" for i in range(len(candidates))])
    ax.set_xlabel("Keyword")
    ax.set_ylabel("Candidate")
    ax.set_title("Domain-keyword occurrence heatmap (candidate × term)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color="white" if v > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="occurrences")
    _save(fig, out)
    return mat


def plot_new_tokens(candidates: list[dict], out: Path, top_n: int = 25) -> list[tuple[str, int]]:
    if len(candidates) < 2:
        return []
    seed_tokens = set(tokenize(candidates[0]["system_prompt"]))
    counts: Counter[str] = Counter()
    for c in candidates[1:]:
        for tok in tokenize(c["system_prompt"]):
            if tok not in seed_tokens:
                counts[tok] += 1
    top = counts.most_common(top_n)
    if not top:
        return []
    labels = [t for t, _ in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(labels) + 1)))
    ax.barh(range(len(labels)), values, color="C2")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Summed occurrences across mutated candidates")
    ax.set_title(f"Top-{len(labels)} new tokens introduced by GEPA (vs. seed)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    _save(fig, out)
    return top


def plot_iteration_scores(trace: list[dict], out: Path) -> dict[str, list]:
    if not trace:
        return {}
    it = [e["i"] for e in trace]
    old_mean = [np.mean(e.get("subsample_scores", [])) if e.get("subsample_scores") else np.nan
                for e in trace]
    new_mean = [np.mean(e.get("new_subsample_scores", [])) if e.get("new_subsample_scores") else np.nan
                for e in trace]
    accepted = [e.get("new_program_idx") is not None for e in trace]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, old_mean, "o-", color="C0", label="parent minibatch mean")
    ax.plot(it, new_mean, "s-", color="C1", label="proposal minibatch mean")
    for i, a in zip(it, accepted, strict=False):
        if a:
            ax.axvline(i, color="C2", alpha=0.25, linewidth=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean score on sampled minibatch (3 ex.)")
    ax.set_title("Parent vs. proposal minibatch scores per iteration "
                 "(green bars = accepted into pool)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, out)
    return {"iteration": it, "parent_mean": old_mean, "proposal_mean": new_mean,
            "accepted": accepted}


def plot_pool_growth(trace: list[dict], num_candidates: int, out: Path) -> list[int]:
    """Size of the candidate pool over iterations."""
    if not trace:
        return []
    pool = [1]  # seed at iter 0
    for e in trace:
        last = pool[-1]
        pool.append(last + (1 if e.get("new_program_idx") is not None else 0))
    pool = pool[:len(trace) + 1]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.step(range(len(pool)), pool, where="post", color="C4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Candidates in pool")
    ax.set_title(f"Candidate pool growth (final size = {num_candidates})")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out)
    return pool


def plot_reflection_lengths(reflection: list[dict], out: Path) -> dict[str, list[int]]:
    if not reflection:
        return {}
    idx = [r.get("call_index", i + 1) for i, r in enumerate(reflection)]
    prompt_chars = [len(json.dumps(r.get("messages", ""))) for r in reflection]
    resp_chars = [len(r.get("response") or "") for r in reflection]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, prompt_chars, "o-", label="reflection prompt chars")
    ax.plot(idx, resp_chars, "s-", label="reflection response chars")
    ax.set_xlabel("Reflection call #")
    ax.set_ylabel("Characters")
    ax.set_title("Reflection LM I/O length per call")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out)
    return {"call_index": idx, "prompt_chars": prompt_chars, "response_chars": resp_chars}


def plot_reflection_cumulative(reflection: list[dict], out: Path) -> list[int]:
    if not reflection:
        return []
    chars = [len(json.dumps(r.get("messages", ""))) + len(r.get("response") or "")
             for r in reflection]
    cum = np.cumsum(chars).tolist()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, len(cum) + 1), cum, "o-", color="C5")
    ax.set_xlabel("Reflection call #")
    ax.set_ylabel("Cumulative chars of reflection I/O")
    ax.set_title("Cumulative reflection-LM context burn")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out)
    return cum


def plot_per_task_heatmap(
    per_task: dict[str, dict[tuple[int, int], dict]],
    val_gold: list[dict],
    num_candidates: int,
    out: Path,
) -> np.ndarray:
    if not per_task or not val_gold:
        return np.zeros((0, 0))
    task_names = sorted(per_task.keys(), key=lambda s: int(s.split("_")[1]))
    mat = np.full((num_candidates, len(task_names)), np.nan)
    for t_idx, tname in enumerate(task_names):
        entries = per_task[tname]
        last_prog = 0
        last_resp: str | None = None
        items = sorted(entries.items())
        task_idx = int(tname.split("_")[1])
        if task_idx >= len(val_gold):
            continue
        gold = int(val_gold[task_idx]["gold_score"])
        for (it, prog), payload in items:
            resp = payload.get("full_assistant_response", "")
            pred = parse_score_response(resp)
            s = score_against_gold(pred, gold)
            for fill_prog in range(last_prog, prog + 1):
                if fill_prog < num_candidates:
                    prev = mat[fill_prog, t_idx]
                    mat[fill_prog, t_idx] = s if np.isnan(prev) else prev
            last_prog = prog + 1
            last_resp = resp
        # Forward-fill the most recent best answer to the final candidate index
        if last_prog < num_candidates and last_resp is not None:
            pred = parse_score_response(last_resp)
            s = score_against_gold(pred, gold)
            for fill_prog in range(last_prog, num_candidates):
                if np.isnan(mat[fill_prog, t_idx]):
                    mat[fill_prog, t_idx] = s
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(task_names) + 3), 0.45 * num_candidates + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(num_candidates))
    ax.set_yticklabels([f"c{i}" for i in range(num_candidates)])
    ax.set_xlabel("Valset task")
    ax.set_ylabel("Candidate")
    ax.set_title("Per-task best-output score (candidate × valset task)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="score (0-1)")
    _save(fig, out)
    return mat


# --- Report assembly --------------------------------------------------------


TEX_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage[T1]{fontenc}

\title{GEPA Prompt Evolution Analysis\\\large Run: \texttt{REPLACE_RUN_NAME}}
\author{Auto-generated from \texttt{reports/generate\_report.py}}
\date{REPLACE_DATE}

\begin{document}
\maketitle

\section{Overview}
This report inspects a single GEPA run that optimized the system prompt for a
Bitcoin forward-looking sentiment scoring task (1--15 scale). GEPA
(reflective prompt optimization) iteratively proposes a new prompt by having a
reflection LM read trajectories and evaluator feedback on a minibatch of
training examples, then accepts the proposal if it improves on the parent
candidate on the minibatch~\cite{gepa}. Related prompt-optimization
approaches include APE~\cite{ape}, PromptBreeder~\cite{promptbreeder}, and
DSPy/COPRO~\cite{dspy}.

\section{Run summary}
\begin{center}
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Candidates (final pool size) & REPLACE_N_CANDIDATES \\
Iterations logged            & REPLACE_N_ITERS \\
Reflection LM calls          & REPLACE_N_REFLECT \\
Seed prompt words            & REPLACE_SEED_WORDS \\
Best candidate words         & REPLACE_BEST_WORDS \\
Word-count growth factor     & REPLACE_GROWTH \\
Jaccard(best, seed)          & REPLACE_JACCARD \\
\bottomrule
\end{tabular}
\end{center}

\section{Prompt length evolution}
GEPA reflective mutation typically \emph{grows} prompts by appending
structured guidance (headers, rules, worked examples). Figure~\ref{fig:len}
shows the word and character count per candidate.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/prompt_length.pdf}
\caption{Prompt length per candidate (words, chars).}
\label{fig:len}
\end{figure}

\section{Lexical drift from seed}
Jaccard token overlap measures how much each candidate departs from the seed's
vocabulary. Low values indicate the reflection LM has introduced a substantial
new vocabulary, consistent with instruction-densification.
\begin{figure}[H]\centering
\includegraphics[width=0.85\linewidth]{figures/jaccard_vs_seed.pdf}
\caption{Jaccard(candidate tokens, seed tokens).}
\end{figure}

\section{Structural features over time}
Prompt optimization often converges on a \emph{format}: headers, bullet
lists, numbered steps, tables, and fenced code blocks for output
specification. Figure~\ref{fig:struct} tracks these structural features per
candidate.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/structural_features.pdf}
\caption{Structural feature counts per candidate.}
\label{fig:struct}
\end{figure}

\section{Domain-keyword heatmap}
A fixed lexicon of Bitcoin-sentiment domain terms, tracked per candidate,
exposes which semantic axes GEPA emphasizes over iterations.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/keyword_heatmap.pdf}
\caption{Domain-keyword occurrence heatmap (candidate $\times$ term).}
\end{figure}

\section{New tokens introduced by GEPA}
Tokens present in at least one mutated candidate but absent from the seed.
This is a first-order readout of what the reflection LM is \emph{adding}.
\begin{figure}[H]\centering
\includegraphics[width=0.9\linewidth]{figures/new_tokens.pdf}
\caption{Top new tokens introduced vs.\ seed.}
\end{figure}

\section{Iteration dynamics}
GEPA's acceptance criterion compares the proposed candidate's mean score on
the sampled minibatch against the parent's mean score. Vertical bars mark
accepted proposals.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/iteration_scores.pdf}
\caption{Parent vs.\ proposal minibatch scores.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.85\linewidth]{figures/pool_growth.pdf}
\caption{Candidate pool growth (Pareto-admissible set size).}
\end{figure}

\section{Per-task score heatmap}
For every accepted candidate, GEPA stores the best model output per valset
task. Re-scoring these outputs against the gold labels exposes which examples
a new prompt fixed or broke, which is more informative than the aggregate
valset score alone.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/per_task_heatmap.pdf}
\caption{Per-task best-output score per candidate.}
\end{figure}

\section{Reflection LM cost}
The reflection LM consumes a large prompt (full trajectory + evaluator
feedback) and returns a new candidate plus rationale. These plots expose how
context size grows across calls.
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/reflection_length.pdf}
\caption{Reflection LM prompt and response lengths.}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.85\linewidth]{figures/reflection_cumulative.pdf}
\caption{Cumulative reflection-LM context burn.}
\end{figure}

\begin{thebibliography}{9}
\bibitem{gepa} GEPA: Reflective Prompt Optimization (Python package,
  \url{https://github.com/gepa-ai/gepa}).
\bibitem{ape} Zhou et al., Large Language Models Are Human-Level Prompt
  Engineers (APE), 2022.
\bibitem{promptbreeder} Fernando et al., PromptBreeder: Self-Referential
  Self-Improvement via Prompt Evolution, 2023.
\bibitem{dspy} Khattab et al., DSPy: Compiling Declarative Language Model
  Calls into Self-Improving Pipelines, 2024.
\end{thebibliography}

\end{document}
"""


def write_tex(path: Path, substitutions: dict[str, str]) -> None:
    text = TEX_TEMPLATE
    for key, val in substitutions.items():
        text = text.replace(key, val)
    path.write_text(text, encoding="utf-8")


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GEPA run analysis report.")
    parser.add_argument("--run-dir", required=True,
                        help="Path to a GEPA run_dir containing candidates.json etc.")
    parser.add_argument("--data", default="data/train/articles.jsonl",
                        help="Training JSONL used to reconstruct the valset for per-task scoring.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write figures and report. Defaults to reports/<run_name>.")
    parser.add_argument("--keywords", nargs="*", default=None,
                        help="Override the default keyword list.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    keywords = args.keywords if args.keywords else DEFAULT_KEYWORDS

    out_root = Path(args.output_dir) if args.output_dir else Path("reports") / run_dir.name
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    global _COMBINED_PDF
    _COMBINED_PDF = PdfPages(out_root / "report.pdf")

    run = load_run(run_dir)
    val_gold = load_val_gold(Path(args.data))
    cands = run["candidates"]
    trace = run["trace"]
    refl = run["reflection"]
    per_task = run["per_task_outputs"]

    if not cands:
        raise SystemExit(f"No candidates.json found in {run_dir}")

    metrics: dict[str, Any] = {"run_dir": str(run_dir), "num_candidates": len(cands)}

    metrics["prompt_length"] = plot_prompt_length(cands, fig_dir / "prompt_length.pdf")
    metrics["jaccard_vs_seed"] = plot_jaccard_vs_seed(cands, fig_dir / "jaccard_vs_seed.pdf")
    metrics["structural"] = plot_structural(cands, fig_dir / "structural_features.pdf")
    kw_mat = plot_keyword_heatmap(cands, keywords, fig_dir / "keyword_heatmap.pdf")
    metrics["keywords"] = {"labels": keywords, "matrix": kw_mat.tolist()}
    metrics["new_tokens"] = plot_new_tokens(cands, fig_dir / "new_tokens.pdf")

    if trace:
        metrics["iteration_scores"] = plot_iteration_scores(trace, fig_dir / "iteration_scores.pdf")
        metrics["pool_growth"] = plot_pool_growth(trace, len(cands), fig_dir / "pool_growth.pdf")

    if refl:
        metrics["reflection_lengths"] = plot_reflection_lengths(
            refl, fig_dir / "reflection_length.pdf")
        metrics["reflection_cumulative"] = plot_reflection_cumulative(
            refl, fig_dir / "reflection_cumulative.pdf")

    if per_task and val_gold:
        metrics["per_task_matrix"] = plot_per_task_heatmap(
            per_task, val_gold, len(cands), fig_dir / "per_task_heatmap.pdf"
        ).tolist()

    # Emit metrics summary JSON
    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    # Emit LaTeX source
    seed_words = word_count(cands[0]["system_prompt"])
    best_words = max(word_count(c["system_prompt"]) for c in cands)
    growth = best_words / seed_words if seed_words else float("nan")
    best_jaccard = max(jaccard(cands[0]["system_prompt"], c["system_prompt"])
                       for c in cands[1:]) if len(cands) > 1 else 1.0
    write_tex(
        out_root / "report.tex",
        {
            "REPLACE_RUN_NAME": run_dir.name.replace("_", r"\_"),
            "REPLACE_DATE": __import__("datetime").date.today().isoformat(),
            "REPLACE_N_CANDIDATES": str(len(cands)),
            "REPLACE_N_ITERS": str(len(trace)),
            "REPLACE_N_REFLECT": str(len(refl)),
            "REPLACE_SEED_WORDS": str(seed_words),
            "REPLACE_BEST_WORDS": str(best_words),
            "REPLACE_GROWTH": f"{growth:.2f}x",
            "REPLACE_JACCARD": f"{best_jaccard:.2f}",
        },
    )

    # Close the multi-page combined PDF
    _COMBINED_PDF.close()

    print(f"[INFO] Report written to: {out_root}")
    print(f"[INFO]   figures:      {fig_dir}")
    print(f"[INFO]   report.tex:   {out_root / 'report.tex'}")
    print(f"[INFO]   report.pdf:   {out_root / 'report.pdf'}")
    print(f"[INFO]   metrics.json: {out_root / 'metrics.json'}")


if __name__ == "__main__":
    main()
