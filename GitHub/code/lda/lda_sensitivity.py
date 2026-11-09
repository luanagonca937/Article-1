"""
Sensitivity analysis of the LDA model used in the article (answers the reviewer's points on K, seeds,
the composite index and the denominator of the guideline analysis).

It rebuilds exactly the same corpus and document-term matrix as lda_bibliometric_analysis_v2.py and:
1) refits LDA for K = 3..6 with three new random seeds and records log perplexity, coherence and the
   combined criterion (-log perplexity + 10 x coherence) -> which K wins for each seed;
2) compares the K = 4 model of each seed with the reference model (seed 1234, saved results): topic matching by
   top-10 word overlap and stability of the set of documents in the digital transformation topic;
3) decomposes the composite index into its three parts and recomputes the ranking under alternative
   weightings (coherence normalised by its maximum; equal weights);
4) counts the Hoshin Kanri guideline expressions in the whole corpus (1,184 and 1,161 documents),
   not only in the digital transformation topic (Topic 1, 502 articles, since V33).

Output: results_v2_corrigido/12_sensitivity_analysis.xlsx
Usage:  python lda_sensitivity.py
"""
import ast
import contextlib
import inspect
import io
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation

import lda_bibliometric_analysis_v2 as base

SEEDS = [1, 42, 2024]          # new seeds; the reference seed 1234 is read from the saved results
K_RANGE = range(3, 8)         # perplexity rises and coherence falls steadily beyond K = 3
REFERENCE_SEED = 1234
REF_K = 3                      # K selected after the IEEE record gained its abstract (02 Nov 2026)
OUTPUT = base.config.results_dir / "12_sensitivity_analysis.xlsx"


def build_corpus():
    """Same steps as base.main(): load .bib files, deduplicate, preprocess, DTM."""
    with contextlib.redirect_stdout(io.StringIO()):
        frames = [base.load_bib_file_complete(str(f)) for f in sorted(base.config.base_dir.glob("*.bib"))]
        names = [f.name for f in sorted(base.config.base_dir.glob("*.bib"))]
        kept = [(df, n) for df, n in zip(frames, names) if not df.empty]
        M, *_ = base.analyze_duplicates([d for d, _ in kept], [n for _, n in kept])
        tokens, valid = base.preprocess_abstracts(M["ABSTRACT"].fillna("").tolist())
        dtm, vectorizer, feature_names = base.create_dtm_simple(tokens)
    return M, M.iloc[valid].reset_index(drop=True), dtm, feature_names


def fit(dtm, k, seed, iterations):
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=iterations,
                                    n_jobs=1, learning_method="batch")
    doc_topic = lda.fit_transform(dtm)
    return lda, doc_topic


def top_words(lda, feature_names, n=10):
    return [[feature_names[i] for i in comp.argsort()[-n:][::-1]] for comp in lda.components_]


def composite_parts(doc_topic, words, dtm, feature_names):
    """The three parts of the composite index exactly as in calculate_topic_priority()."""
    dominant = np.argmax(doc_topic, axis=1)
    counts = np.bincount(dominant, minlength=doc_topic.shape[1])
    avg_prob = doc_topic.mean(axis=0)
    coherence = np.array([base.calculate_topic_coherence_for_topic(feature_names, dtm, w[:5]) for w in words])
    return counts, avg_prob, coherence


def keyword_patterns():
    """Guideline dictionary read from the main script, so both use the same expressions."""
    source = inspect.getsource(base.run_hoshin_kanri_analysis)
    literal = re.search(r"keywords_map = (\{.*?\n    \})", source, flags=re.S).group(1)
    keywords_map = ast.literal_eval(literal)
    return {d: re.compile(r"\b(?:" + "|".join(kws) + r")\b") for d, kws in keywords_map.items()}


def guideline_text(title, abstract):
    text = unicodedata.normalize("NFKD", f"{base.normalize_text(title)} {base.normalize_text(abstract)}".lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[-/]", " ", text))


def guideline_shares(frame, patterns):
    texts = [guideline_text(t, a) for t, a in zip(frame["TITLE"].fillna(""), frame["ABSTRACT"].fillna(""))]
    hits = pd.DataFrame({d: [int(bool(p.search(t))) for t in texts] for d, p in patterns.items()})
    return hits


def main():
    M, M_processed, dtm, feature_names = build_corpus()
    print(f"Corpus: {len(M)} records; {dtm.shape[0]} documents in the DTM; {dtm.shape[1]} terms")

    # 1) K x seed grid (500 iterations, as in the K selection of the main script). The reference seed is
    #    not refitted: its values come from 02_k_metrics.xlsx, written by the main script.
    ref_metrics = pd.read_excel(base.config.results_dir / "02_k_metrics.xlsx")
    grid = [{"Seed": REFERENCE_SEED, "K": int(r.K), "Log_Perplexity": r.Log_Perplexity,
             "Coherence": r.Topic_Coherence, "Combined": r.Composite_Score} for r in ref_metrics.itertuples()
            if r.K in K_RANGE]
    k4_models = {}
    for seed in SEEDS:
        for k in K_RANGE:
            t0 = time.time()
            lda, doc_topic = fit(dtm, k, seed, 500)
            log_perp = base.calculate_log_perplexity(lda, dtm)
            coherence = base.calculate_topic_coherence(lda, feature_names, dtm, 10)
            grid.append({"Seed": seed, "K": k, "Log_Perplexity": log_perp, "Coherence": coherence,
                         "Combined": -log_perp + 10 * coherence})
            if k == REF_K:
                k4_models[seed] = (lda, doc_topic)
            print(f"seed={seed} K={k} logperp={log_perp:.4f} coh={coherence:.4f} ({time.time() - t0:.0f} s)")
    grid = pd.DataFrame(grid)
    best = grid.loc[grid.groupby("Seed")["Combined"].idxmax(), ["Seed", "K", "Combined"]]
    best = best.rename(columns={"K": "Best_K"})
    best_perp = grid.loc[grid.groupby("Seed")["Log_Perplexity"].idxmin(), ["Seed", "K"]].rename(columns={"K": "K_min_perplexity"})
    best_coh = grid.loc[grid.groupby("Seed")["Coherence"].idxmax(), ["Seed", "K"]].rename(columns={"K": "K_max_coherence"})
    best = best.merge(best_perp, on="Seed").merge(best_coh, on="Seed")
    gap = grid[grid.K.isin([REF_K, REF_K + 1])].pivot(index="Seed", columns="K", values="Combined")
    best["Combined_K3_minus_K4"] = best["Seed"].map(gap[REF_K] - gap[REF_K + 1])

    # 2) Stability of K = 4: the reference model (seed 1234, 2,000 iterations) is read from 03_lda_results.xlsx
    ref_docs = pd.read_excel(base.config.results_dir / "03_lda_results.xlsx", sheet_name="Documents with Topics")
    ref_words = [w.split(", ") for w in pd.read_excel(base.config.results_dir / "03_lda_results.xlsx",
                                                      sheet_name="LDA Words")["Top_Words"]]
    prio = pd.read_excel(base.config.results_dir / "COMPLETE_LDA_REPORT.xlsx", sheet_name="Topics_Priority")
    prio = prio.set_index("Topic").sort_index()
    ref_counts = prio["Documents"].values
    ref_prob = prio["Avg_Probability"].values
    ref_coh = prio["Topic_Coherence"].values
    ref_dt = int(np.argmax([("digital" in w) + ("implementation" in w) for w in ref_words]))
    ref_set = set(np.where(ref_docs["main_topic"].values == ref_dt + 1)[0])

    stability, composite_rows = [], []
    for seed, (lda, doc_topic) in k4_models.items():
        words = top_words(lda, feature_names)
        overlap = np.array([[len(set(a) & set(b)) / 10 for b in words] for a in ref_words])
        rows, cols = linear_sum_assignment(-overlap)
        match = dict(zip(rows, cols))
        dt = match[ref_dt]
        dt_set = set(np.where(np.argmax(doc_topic, axis=1) == dt)[0])
        counts, avg_prob, coherence = composite_parts(doc_topic, words, dtm, feature_names)
        freq_pts = counts / counts.max() * 40
        prob_pts = avg_prob / avg_prob.max() * 35
        coh_pts = np.minimum(coherence * 25, 25)
        score = freq_pts + prob_pts + coh_pts
        stability.append({
            "Seed": seed,
            "Mean_top10_overlap_with_reference": overlap[rows, cols].mean(),
            "Min_top10_overlap_with_reference": overlap[rows, cols].min(),
            "DT_topic_top_words": ", ".join(words[dt]),
            "DT_topic_documents": len(dt_set),
            "Jaccard_DT_documents_vs_reference": len(dt_set & ref_set) / len(dt_set | ref_set),
            "DT_topic_rank_in_composite_index": int((score > score[dt]).sum() + 1),
        })
        for t in range(REF_K):
            composite_rows.append({"Seed": seed, "Topic_matched_to_reference": int(rows[list(cols).index(t)]) + 1,
                                   "Top_words": ", ".join(words[t][:6]), "Documents": int(counts[t]),
                                   "Avg_probability": avg_prob[t], "Coherence_top5": coherence[t],
                                   "Frequency_points": freq_pts[t], "Probability_points": prob_pts[t],
                                   "Coherence_points": coh_pts[t], "Composite": score[t]})
    stability = pd.DataFrame(stability)
    composite = pd.DataFrame(composite_rows)

    # 3) Alternative weightings of the composite index for the reference model
    f = ref_counts / ref_counts.max()
    p = ref_prob / ref_prob.max()
    c = ref_coh / ref_coh.max()
    alternatives = pd.DataFrame({
        "Topic": range(1, REF_K + 1), "Top_words": [", ".join(w[:6]) for w in ref_words],
        "Documents": ref_counts, "Avg_probability": ref_prob, "Coherence_top5": ref_coh,
        "Original_40_35_25(coh_x25)": 40 * f + 35 * p + np.minimum(25 * ref_coh, 25),
        "Coherence_normalised_40_35_25": 40 * f + 35 * p + 25 * c,
        "Equal_weights_normalised": 100 / 3 * (f + p + c),
        "Frequency_only": 100 * f,
    })
    for i, col in enumerate(list(alternatives.columns[5:]), start=1):
        alternatives[f"Rank_{i}"] = alternatives[col].rank(ascending=False).astype(int)

    # 4) Guideline expressions in the whole corpus versus the digital transformation topic
    patterns = keyword_patterns()
    all_hits = guideline_shares(M, patterns)
    lda_hits = guideline_shares(M_processed, patterns)
    dt_hits = lda_hits.iloc[sorted(ref_set)]
    corpus = pd.DataFrame({
        "Guideline": list(patterns),
        "All_records_n": all_hits.sum().values, "All_records_%": (all_hits.mean() * 100).round(2).values,
        "LDA_documents_n": lda_hits.sum().values, "LDA_documents_%": (lda_hits.mean() * 100).round(2).values,
        "Topic2_n": dt_hits.sum().values, "Topic2_%": (dt_hits.mean() * 100).round(2).values,
    })
    summary_rows = []
    for name, hits in [("All records", all_hits), ("LDA documents", lda_hits), ("Topic 2 (reference)", dt_hits)]:
        total = hits.sum(axis=1)
        summary_rows.append({"Universe": name, "Documents": len(hits), "At_least_one_guideline": int((total >= 1).sum()),
                             "Two_or_more": int((total >= 2).sum()), "Max_guidelines_in_one_article": int(total.max()),
                             "Guidelines_never_found": int((hits.sum() == 0).sum()),
                             "Highest_share_%": round(float(hits.mean().max() * 100), 2)})
    corpus_summary = pd.DataFrame(summary_rows)

    expected_corr = pd.DataFrame({"K": list(K_RANGE), "Expected_mean_correlation_-1/(K-1)": [-1 / (k - 1) for k in K_RANGE]})
    with pd.ExcelWriter(OUTPUT) as w:
        grid.to_excel(w, sheet_name="K_by_seed", index=False)
        best.to_excel(w, sheet_name="Best_K_by_seed", index=False)
        stability.to_excel(w, sheet_name="K3_stability", index=False)
        composite.to_excel(w, sheet_name="Composite_parts_by_seed", index=False)
        alternatives.to_excel(w, sheet_name="Composite_alternatives", index=False)
        corpus.to_excel(w, sheet_name="Guidelines_by_universe", index=False)
        corpus_summary.to_excel(w, sheet_name="Guidelines_summary", index=False)
        expected_corr.to_excel(w, sheet_name="Expected_correlation", index=False)
    print(best.to_string(), stability.to_string(), alternatives.to_string(), corpus_summary.to_string(),
          corpus.to_string(), sep="\n\n")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
