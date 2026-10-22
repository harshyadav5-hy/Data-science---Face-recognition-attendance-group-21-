"""Measure how well your ENROLLED students can be told apart, and suggest thresholds.

    python calibrate.py

For every pair of stored face samples it computes the cosine similarity (the number the matcher uses):
  genuine  = two samples of the SAME student      (should be high)
  impostor = samples of DIFFERENT students        (should be low)
and reports how many students would be falsely accepted / rejected at the current MATCH_THRESHOLD.

Honest limits: the samples were all captured in one enrolment session, so genuine scores here are
optimistic (real-life lighting/pose varies more), and the live matcher averages several frames,
which makes real genuine scores higher still.  Use this to find enrolment problems (two students
who look alike to the model, a student with one bad sample) and to sanity-check MATCH_THRESHOLD -
then confirm with the live camera test described in README "Calibrating".
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

import config


def analyse(records, threshold: float = config.MATCH_THRESHOLD) -> dict:
    """records: [(reg_no, name, vector)].  Returns statistics (pure function, unit-tested)."""
    by_id: dict[str, list[int]] = defaultdict(list)
    vectors = []
    for i, (reg_no, _name, vec) in enumerate(records):
        by_id[reg_no].append(i)
        v = np.asarray(vec, np.float32)
        vectors.append(v / max(float(np.linalg.norm(v)), 1e-9))
    out = {"students": len(by_id), "samples": len(vectors), "threshold": threshold,
           "genuine": np.zeros(0), "impostor": np.zeros(0), "worst_pairs": []}
    if len(vectors) < 2:
        return out
    gram = np.asarray(vectors) @ np.asarray(vectors).T
    label = np.empty(len(vectors), object)
    for reg_no, idx in by_id.items():
        label[idx] = reg_no
    upper = np.triu(np.ones_like(gram, bool), 1)
    same = (label[:, None] == label[None, :]) & upper
    diff = (label[:, None] != label[None, :]) & upper
    out["genuine"], out["impostor"] = gram[same], gram[diff]

    # per student pair: the highest similarity between any of their samples (what a lookalike would score)
    ids = sorted(by_id)
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pairs.append((float(gram[np.ix_(by_id[ids[a]], by_id[ids[b]])].max()), ids[a], ids[b]))
    out["worst_pairs"] = sorted(pairs, reverse=True)[:5]

    if len(out["impostor"]):
        imp_max = float(out["impostor"].max())
        out["far"] = float((out["impostor"] >= threshold).mean())
        out["suggested"] = float(min(0.70, max(0.36, imp_max + 0.03)))
    if len(out["genuine"]):
        out["frr"] = float((out["genuine"] < threshold).mean())
    return out


def _pct(values: np.ndarray, qs=(5, 50)) -> str:
    return "n/a" if not len(values) else "  ".join(f"p{q}={np.percentile(values, q):.2f}" for q in qs)


def report(stats: dict) -> str:
    lines = [f"Students: {stats['students']}   samples: {stats['samples']}   current MATCH_THRESHOLD: {stats['threshold']:.2f}", ""]
    g, i = stats["genuine"], stats["impostor"]
    if not len(g) or not len(i):
        return "\n".join(lines + ["Need at least 2 students with 2+ samples each to analyse."])
    lines += [f"Same student     (n={len(g):>6}): min={g.min():.2f}  {_pct(g)}",
              f"Different people (n={len(i):>6}): max={i.max():.2f}  {_pct(i, (50, 95, 99))}", "",
              f"At {stats['threshold']:.2f}:  false accepts {stats['far']:.3%} of impostor pairs,  "
              f"false rejects {stats['frr']:.2%} of same-student pairs", ""]
    if stats["worst_pairs"]:
        lines.append("Most similar pairs of DIFFERENT students (check these two are not confused):")
        lines += [f"   {a} vs {b}: {s:.2f}" for s, a, b in stats["worst_pairs"]]
        lines.append("")
    sug, imp_max = stats["suggested"], float(i.max())
    if stats["far"] > 0:
        lines.append(f"WARNING: some different-student pairs already score >= {stats['threshold']:.2f}. "
                     f"Raise MATCH_THRESHOLD to about {sug:.2f} and/or re-enroll the students listed above.")
    elif imp_max <= stats["threshold"] - 0.10:
        lines.append(f"Your students are well separated (highest score between different students: {imp_max:.2f}); "
                     f"{stats['threshold']:.2f} is safe. Do NOT lower it below ~0.36 unless real students are often 'Unknown'.")
    else:
        lines.append(f"No false accepts at {stats['threshold']:.2f}, but the margin is small (highest score between "
                     f"different students: {imp_max:.2f}). Consider {sug:.2f} and re-check the pairs above.")
    if stats["frr"] > 0.05:
        lines.append(f"NOTE: {stats['frr']:.0%} of same-student pairs score below the threshold: some enrolments are poor "
                     f"(blur, glare, one odd pose). Re-enroll those students in good light.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    args = parser.parse_args()
    from database import Database
    records = Database().load_embeddings(config.EMBEDDING_MODEL_TAG)
    if not records:
        print("No enrolled students found.")
        return 1
    print(report(analyse(records, args.threshold)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
