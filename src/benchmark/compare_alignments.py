"""
Compare two utterance-level alignment TSVs produced by align_utterances.sh.

Both files must contain the same utterances in the same order (same audio + text,
different alignment systems).  The script computes per-utterance boundary
differences and reports PTEM-style summary statistics plus a ranked list of the
clips with the largest disagreement, with paths to the audio files for listening.

Outputs (written to --output_dir if provided):
  comparison_report.txt   full text report
  score_distributions.png histogram + KDE of segment scores for both systems

Usage
-----
    python src/benchmark/compare_alignments.py \\
        --ref        data/wip_benedetti/results/benedetti_aligned.tsv \\
        --hyp        data/wip_benedetti_omni/results/benedetti_aligned.tsv \\
        --ref_label  wav2vec2 \\
        --hyp_label  omniASR \\
        --clips_ref  data/wip_benedetti/clips \\
        --clips_hyp  data/wip_benedetti_omni/clips \\
        --output_dir data/wip_benedetti_omni/benchmark \\
        --top 15 \\
        --collar 0.0
"""

import argparse
import io
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def _clip_path(clips_dir: str, sample_id: str) -> str:
    if not clips_dir:
        return ""
    path = os.path.join(clips_dir, sample_id + ".wav")
    return path if os.path.isfile(path) else path + "  [NOT FOUND]"


def _build_report(args, ref, hyp, start_diffs, end_diffs, te_list) -> str:
    n = len(ref)
    collar = args.collar
    ref_label = args.ref_label
    hyp_label = args.hyp_label

    buf = io.StringIO()

    def p(line=""):
        buf.write(line + "\n")

    p(f"Alignment comparison — {n} utterances")
    p(f"  ref [{ref_label}] : {args.ref}")
    p(f"  hyp [{hyp_label}] : {args.hyp}")
    p(f"  collar: {collar:.3f} s")
    p()
    p("Boundary Time-Error (PTEM-style)")
    p(f"  MEDIAN start diff   {statistics.median(start_diffs):6.3f} s")
    p(f"  MEDIAN end diff     {statistics.median(end_diffs):6.3f} s")
    p(f"  MEDIAN total (TE)   {statistics.median(te_list):6.3f} s")
    p(f"  MEAN   total (TE)   {statistics.mean(te_list):6.3f} s")
    p(f"  MAE    start        {sum(start_diffs)/n:6.3f} s")
    p(f"  MAE    end          {sum(end_diffs)/n:6.3f} s")

    thresholds = [0.1, 0.2, 0.5, 1.0]
    p()
    p("  |start diff| within threshold:")
    for t in thresholds:
        pct = sum(1 for d in start_diffs if d <= t) / n * 100
        p(f"    <= {t:.1f} s : {pct:.1f}%")
    p("  |end diff| within threshold:")
    for t in thresholds:
        pct = sum(1 for d in end_diffs if d <= t) / n * 100
        p(f"    <= {t:.1f} s : {pct:.1f}%")

    if "Segment_Score" in ref.columns and "Segment_Score" in hyp.columns:
        ref_scores = ref["Segment_Score"].tolist()
        hyp_scores = hyp["Segment_Score"].tolist()
        thr = -2.0
        p()
        p("Score comparison")
        p(f"  Mean score — {ref_label}: {statistics.mean(ref_scores):.4f}  "
          f"{hyp_label}: {statistics.mean(hyp_scores):.4f}")
        p(f"  Median score — {ref_label}: {statistics.median(ref_scores):.4f}  "
          f"{hyp_label}: {statistics.median(hyp_scores):.4f}")
        p(f"  Std score — {ref_label}: {statistics.stdev(ref_scores):.4f}  "
          f"{hyp_label}: {statistics.stdev(hyp_scores):.4f}")
        p(f"  Utterances below {thr} — {ref_label}: "
          f"{sum(1 for s in ref_scores if s < thr)}  "
          f"{hyp_label}: {sum(1 for s in hyp_scores if s < thr)}")
        # Score proximity: fraction of utterances where both systems agree within 0.5
        n_close = sum(
            1 for r, h in zip(ref_scores, hyp_scores) if abs(r - h) <= 0.5
        )
        p(f"  Scores within 0.5 of each other: {n_close}/{n} ({100*n_close/n:.1f}%)")

    ranked = sorted(range(n), key=lambda i: te_list[i], reverse=True)
    top = ranked[: args.top]

    p()
    p(f"Top {args.top} utterances by total boundary error")
    p(f"{'#':>4}  {'|Dstart|':>9}  {'|Dend|':>9}  {'TE':>9}  transcription")
    for rank, idx in enumerate(top, 1):
        txt = str(ref.loc[idx, "Transcription"])[:60]
        p(f"{rank:>4}  {start_diffs[idx]:>9.3f}  {end_diffs[idx]:>9.3f}  "
          f"{te_list[idx]:>9.3f}  {txt}")

    if args.clips_ref or args.clips_hyp:
        p()
        p(f"Clips to review (top {args.top} outliers)")
        for rank, idx in enumerate(top, 1):
            ref_id = str(ref.loc[idx, "Sample_ID"])
            hyp_id = str(hyp.loc[idx, "Sample_ID"])
            p()
            p(f"  [{rank}] TE={te_list[idx]:.3f}s  "
              f"|Dstart|={start_diffs[idx]:.3f}s  |Dend|={end_diffs[idx]:.3f}s")
            p(f"      text : {str(ref.loc[idx, 'Transcription'])}")
            p(f"      {ref_label}  start={float(ref.loc[idx,'Start']):.3f}  "
              f"end={float(ref.loc[idx,'End']):.3f}  "
              f"score={float(ref.loc[idx,'Segment_Score']):.4f}")
            p(f"      {hyp_label}  start={float(hyp.loc[idx,'Start']):.3f}  "
              f"end={float(hyp.loc[idx,'End']):.3f}  "
              f"score={float(hyp.loc[idx,'Segment_Score']):.4f}")
            if args.clips_ref:
                p(f"      ref clip : {_clip_path(args.clips_ref, ref_id)}")
            if args.clips_hyp:
                p(f"      hyp clip : {_clip_path(args.clips_hyp, hyp_id)}")

    return buf.getvalue()


def _save_score_plot(ref_scores, hyp_scores, ref_label, hyp_label, out_path):
    ref_arr = np.array(ref_scores, dtype=float)
    hyp_arr = np.array(hyp_scores, dtype=float)

    # Clip extreme outliers for display only (below -10 distort the axis)
    display_min = max(min(ref_arr.min(), hyp_arr.min()), -10.0)
    bins = np.linspace(display_min, 0.0, 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Segment score distributions", fontsize=13)

    # ---- left panel: overlapping histograms ----
    ax = axes[0]
    ax.hist(np.clip(ref_arr, display_min, 0), bins=bins,
            alpha=0.55, color="#2176AE", label=ref_label, density=True)
    ax.hist(np.clip(hyp_arr, display_min, 0), bins=bins,
            alpha=0.55, color="#E07A5F", label=hyp_label, density=True)
    ax.axvline(-2.0, color="black", linestyle="--", linewidth=1, label="threshold −2.0")
    ax.set_xlabel("Segment score (log-prob per frame)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram")
    ax.legend()

    # ---- right panel: KDE ----
    ax2 = axes[1]
    x_grid = np.linspace(display_min, 0.05, 400)
    for arr, color, label in [
        (ref_arr, "#2176AE", ref_label),
        (hyp_arr, "#E07A5F", hyp_label),
    ]:
        clipped = np.clip(arr, display_min, 0)
        kde = gaussian_kde(clipped, bw_method="scott")
        ax2.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
        ax2.axvline(float(np.median(clipped)), color=color,
                    linestyle=":", linewidth=1.5,
                    label=f"median {label} {np.median(clipped):.3f}")
    ax2.axvline(-2.0, color="black", linestyle="--", linewidth=1, label="threshold −2.0")
    ax2.set_xlabel("Segment score (log-prob per frame)")
    ax2.set_ylabel("Density")
    ax2.set_title("KDE")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    ref = pd.read_csv(args.ref, sep="\t")
    hyp = pd.read_csv(args.hyp, sep="\t")

    if len(ref) != len(hyp):
        print(f"WARNING: row count mismatch — ref {len(ref)}, hyp {len(hyp)}")
    n = min(len(ref), len(hyp))
    ref = ref.iloc[:n].reset_index(drop=True)
    hyp = hyp.iloc[:n].reset_index(drop=True)

    collar = args.collar
    start_diffs, end_diffs, te_list = [], [], []

    for i in range(n):
        d_start = abs(float(ref.loc[i, "Start"]) - float(hyp.loc[i, "Start"]))
        d_end   = abs(float(ref.loc[i, "End"])   - float(hyp.loc[i, "End"]))
        if d_start < collar:
            d_start = 0.0
        if d_end < collar:
            d_end = 0.0
        start_diffs.append(d_start)
        end_diffs.append(d_end)
        te_list.append(d_start + d_end)

    report = _build_report(args, ref, hyp, start_diffs, end_diffs, te_list)
    print(report)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        txt_path = os.path.join(args.output_dir, "comparison_report.txt")
        with open(txt_path, "w") as f:
            f.write(report)
        print(f"Report saved to {txt_path}")

        if "Segment_Score" in ref.columns and "Segment_Score" in hyp.columns:
            img_path = os.path.join(args.output_dir, "score_distributions.png")
            _save_score_plot(
                ref["Segment_Score"].tolist(),
                hyp["Segment_Score"].tolist(),
                args.ref_label,
                args.hyp_label,
                img_path,
            )
            print(f"Score distribution plot saved to {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two alignment TSVs and flag clips with large boundary disagreement"
    )
    parser.add_argument("--ref", required=True, help="reference alignment TSV")
    parser.add_argument("--hyp", required=True, help="hypothesis alignment TSV")
    parser.add_argument("--ref_label", default="ref",
                        help="display name for the reference system (default: ref)")
    parser.add_argument("--hyp_label", default="hyp",
                        help="display name for the hypothesis system (default: hyp)")
    parser.add_argument("--clips_ref", default="", help="directory with ref WAV clips")
    parser.add_argument("--clips_hyp", default="", help="directory with hyp WAV clips")
    parser.add_argument("--output_dir", default="",
                        help="directory to write comparison_report.txt and score_distributions.png")
    parser.add_argument("--top", type=int, default=10,
                        help="number of worst-case utterances to display (default: 10)")
    parser.add_argument("--collar", type=float, default=0.0,
                        help="errors below this threshold (seconds) count as zero (default: 0.0)")
    args = parser.parse_args()
    main(args)
