"""Reproduce the literature-mapping results of Article 1 from the exported BibTeX records.

Steps
1. Build the corpus (16 BibTeX files, title deduplication), fit LDA (K selected from 3 to 8), rank the topics and count
   the guideline expressions (lda_bibliometric_analysis_v2.py).
2. Document-topic correlations and the intertopic map (topic_correlations_and_map.py).
3. With --full: sensitivity across random seeds and guideline stability (lda_sensitivity.py, lda_guideline_stability.py).
   These take about one hour more.
4. Draw the manuscript figures (code/figures/make_figures.py).
5. Compare the main values with those reported in the article (data/EXPECTED_RESULTS.json).

The exported records are not redistributed. Place the BibTeX files exported with the search protocol in
data/SEARCH_PROTOCOL.md into data/raw_bib/ before running. Outputs go to GENERATED_RESULTS/ next to this script.
The questionnaire and interview materials are not part of this public package because their inputs are confidential or retained in the controlled research archive (see README.md).
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LDA = HERE / "code" / "lda"
RAW = HERE / "data" / "raw_bib"
OUT = HERE / "GENERATED_RESULTS"
RUN = OUT / "run"
RES = RUN / "results_v2_corrigido"


def run(script, *args):
    print("\n>", script.name, *args, flush=True)
    with open(OUT / f"log_{script.stem}.txt", "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(script), *map(str, args)], cwd=RUN, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env={**__import__("os").environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"})


def main(full):
    bibs = sorted(RAW.glob("*.bib"))
    if not bibs:
        sys.exit(f"No .bib files in {RAW}. See data/SEARCH_PROTOCOL.md.")
    RUN.mkdir(parents=True, exist_ok=True)
    for f in list(LDA.glob("*.py")) + bibs:     # the LDA script reads the .bib files of its own folder
        shutil.copy2(f, RUN / f.name)
    run(RUN / "lda_bibliometric_analysis_v2.py")
    run(RUN / "topic_correlations_and_map.py")
    if full:
        run(RUN / "lda_sensitivity.py")
        run(RUN / "lda_guideline_stability.py")
    run(HERE / "code" / "figures" / "make_figures.py", RES, OUT / "figures")
    verify()


def verify():
    expected = json.loads((HERE / "data" / "EXPECTED_RESULTS.json").read_text(encoding="utf-8"))
    log = (OUT / "log_lda_bibliometric_analysis_v2.txt").read_text(encoding="utf-8", errors="replace")
    prio = pd.read_excel(RES / "COMPLETE_LDA_REPORT.xlsx", sheet_name="Topics_Priority").sort_values("Topic")
    gl = pd.read_excel(RES / "Hoshin Kanri" / "table_guidelines_complete.xlsx").set_index("Guideline")
    got = {
        "unique_records": len(pd.read_excel(RES / "base_bibliometrix_COMPLETE.xlsx")),
        "modelled_documents": int(re.search(r"Documents after processing: (\d+)", log).group(1)),
        "rejected_documents": int(re.search(r"Documents rejected: (\d+)", log).group(1)),
        "terms": int(re.search(r"DTM dimensions: \(\d+, (\d+)\)", log).group(1)),
        "selected_K": int(re.search(r"Optimal K: (\d+)", log).group(1)),
        "documents_per_topic": [int(x) for x in prio["Documents"]],
        "composite_index": [round(float(x), 2) for x in prio["Composite_Score"]],
        "topic1_D03_percent": round(float(gl.loc["D03", "Percentage"]), 2),
        "topic1_D01_percent": round(float(gl.loc["D01", "Percentage"]), 2),
    }
    sens = RES / "12_sensitivity_analysis.xlsx"
    if sens.exists():
        s = pd.read_excel(sens, sheet_name="Guidelines_summary").set_index("Universe")
        got["whole_corpus_records_with_a_guideline"] = int(s.loc["All records", "At_least_one_guideline"])
    report, tally = [], {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for key, value in expected.items():
        status = "SKIPPED" if key not in got else ("PASS" if got[key] == value else "FAIL")
        tally[status] += 1
        report.append({"value": key, "expected": value, "obtained": got.get(key), "status": status})
        print(f"{status}  {key}: expected {value}, obtained {got.get(key)}")
    (OUT / "VERIFICATION_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tally)
    sys.exit(1 if tally["FAIL"] else 0)


if __name__ == "__main__":
    if "--verify-only" in sys.argv:
        verify()
    else:
        main("--full" in sys.argv)
