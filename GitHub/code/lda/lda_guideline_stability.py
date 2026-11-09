"""
Stability of the guideline shares in the digital transformation topic (Topic 1 since V33) across random seeds.

For the reference model (seed 1234, saved results) and for the K = 3 models of seeds 1, 42 and 2024
(500 iterations, as in lda_sensitivity.py), the digital transformation topic is identified by top-word overlap
with the reference topic, and the guideline dictionary of the main script is applied to the titles and
abstracts of its documents. Also exports the full dictionary and the matched passages, used to document
positive and false-positive examples.

Output: results_v2_corrigido/13_guideline_stability.xlsx
Usage:  python lda_guideline_stability.py
"""
import re
import time

import numpy as np
import pandas as pd

import lda_bibliometric_analysis_v2 as base
from lda_sensitivity import build_corpus, fit, guideline_text, keyword_patterns, top_words

SEEDS = [1, 42, 2024]
OUTPUT = base.config.results_dir / "13_guideline_stability.xlsx"


def main():
    M, M_processed, dtm, feature_names = build_corpus()
    patterns = keyword_patterns()
    texts = [guideline_text(t, a) for t, a in zip(M_processed["TITLE"].fillna(""), M_processed["ABSTRACT"].fillna(""))]
    hits = pd.DataFrame({d: [int(bool(p.search(t))) for t in texts] for d, p in patterns.items()})

    res = base.config.results_dir
    ref_docs = pd.read_excel(res / "03_lda_results.xlsx", sheet_name="Documents with Topics")
    ref_words = [w.split(", ") for w in pd.read_excel(res / "03_lda_results.xlsx", sheet_name="LDA Words")["Top_Words"]]
    ref_dt = int(np.argmax([("digital" in w) + ("implementation" in w) for w in ref_words]))
    sets = {"1234 (reference)": np.where(ref_docs["main_topic"].values == ref_dt + 1)[0]}

    for seed in SEEDS:
        t0 = time.time()
        lda, doc_topic = fit(dtm, 3, seed, 500)
        words = top_words(lda, feature_names)
        dt = int(np.argmax([len(set(w) & set(ref_words[ref_dt])) for w in words]))
        sets[str(seed)] = np.where(np.argmax(doc_topic, axis=1) == dt)[0]
        print(f"seed={seed}: topic {dt + 1}, {len(sets[str(seed)])} documents ({time.time() - t0:.0f} s)")

    rows = []
    for d in patterns:
        row = {"Guideline": d}
        for name, idx in sets.items():
            row[f"{name}_n"] = int(hits.iloc[idx][d].sum())
            row[f"{name}_%"] = round(float(hits.iloc[idx][d].mean() * 100), 2)
        pct = [row[f"{name}_%"] for name in sets]
        row["Min_%"], row["Max_%"] = min(pct), max(pct)
        rows.append(row)
    shares = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "Run": name, "Documents": len(idx),
        "At_least_one_%": round(float((hits.iloc[idx].sum(axis=1) >= 1).mean() * 100), 2),
        "Two_or_more": int((hits.iloc[idx].sum(axis=1) >= 2).sum()),
        "Max_guidelines": int(hits.iloc[idx].sum(axis=1).max()),
        "Never_found": int((hits.iloc[idx].sum() == 0).sum()),
        "Top_guideline": hits.iloc[idx].mean().idxmax(),
        "Top_share_%": round(float(hits.iloc[idx].mean().max() * 100), 2),
        "Second_guideline": hits.iloc[idx].mean().sort_values(ascending=False).index[1],
    } for name, idx in sets.items()])

    # dictionary and matched passages (for positive and false-positive examples)
    source = re.search(r"keywords_map = (\{.*?\n    \})", __import__("inspect").getsource(base.run_hoshin_kanri_analysis),
                       flags=re.S).group(1)
    dictionary = pd.DataFrame([{"Guideline": d, "Expressions (regular expressions, whole words)": " | ".join(v)}
                               for d, v in __import__("ast").literal_eval(source).items()])
    matches = []
    for i in sets["1234 (reference)"]:
        for d, p in patterns.items():
            m = p.search(texts[i])
            if m:
                matches.append({"Guideline": d, "Match": m.group(0), "Title": M_processed["TITLE"].iloc[i],
                                "Context": texts[i][max(0, m.start() - 120): m.end() + 120]})
    matches = pd.DataFrame(matches)

    with pd.ExcelWriter(OUTPUT) as w:
        summary.to_excel(w, sheet_name="Summary_by_run", index=False)
        shares.to_excel(w, sheet_name="Guideline_shares_by_run", index=False)
        dictionary.to_excel(w, sheet_name="Dictionary", index=False)
        matches.to_excel(w, sheet_name="Matches_reference_Topic2", index=False)
    print(summary.to_string(), shares[["Guideline", "Min_%", "Max_%"] + [c for c in shares if c.endswith("_%") and c[0].isdigit()]].to_string(), sep="\n\n")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
