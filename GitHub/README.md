# Reproduction package: Hoshin Kanri guidelines in agro-industrial cooperatives

Code and processed data for the article *Adapting Hoshin Kanri Implementation Guidelines in Agro-Industrial Cooperatives with Poultry Operations: Governance, Member Participation and Digital Transformation*.

The study is exploratory. An auxiliary literature mapping (LDA topic model and a dictionary of guideline expressions) precedes two empirical stages: expert review and pretest of a questionnaire derived from the twenty Hoshin Kanri guidelines of Giordani da Silveira et al. (2017), and its application, with semi-structured interviews, in two agro-industrial cooperatives with poultry operations. Participants are identified only by codes (E1–E5, C1–C2).

The package runs from any folder. Inputs are read from `data/` and outputs are written to `GENERATED_RESULTS/`, both next to the script.

## What is and is not included

| Included | Not included, and why |
|---|---|
| Literature-mapping code (`code/lda/`, `code/figures/`), MIT license | Exported BibTeX records: subject to the license terms of ScienceDirect, Scopus, Web of Science and IEEE Xplore |
| Search protocol and the lists of the 1,272 unique and 1,249 modelled records, without abstracts (`data/`) | Abstracts and passages of published articles: copyrighted |
| Archived literature-mapping result tables reported in the article (`archived_results/`) | Interview recordings and transcripts, questionnaire exports, qualitative-analysis scripts containing source-language evidence, and the key linking codes to participants: confidential or retained in the controlled research archive |

## Run

Requires Python 3.10 or newer.

1. Repeat the search in `data/SEARCH_PROTOCOL.md` and export the records as BibTeX with abstracts.
2. Put the 16 `.bib` files in `data/raw_bib/`.
3. Run:

| System | Command |
|---|---|
| Windows | `run_windows.bat` |
| Linux / macOS | `bash run_linux_mac.sh` |
| Any (libraries already installed) | `python reproduce_article1.py` |

The script builds the corpus, fits LDA (K selected from 3 to 8 with the combined criterion), ranks the topics, counts the guideline expressions, computes the topic correlations and the intertopic map, draws the figures and compares the main values with those in the article (`data/EXPECTED_RESULTS.json`). It prints `PASS`, `FAIL` or `SKIPPED` for each value and saves `GENERATED_RESULTS/VERIFICATION_REPORT.json`.

Add `--full` to run also the sensitivity analysis across random seeds and the guideline stability analysis (about one hour more). Without `--full`, the value that depends on them is `SKIPPED`.

The LDA seed is fixed (1234; seeds 1, 42 and 2024 in the sensitivity analysis). Because the databases add records over time, a new export can differ from the one used in the article (1 and 2 November 2026), and some values may then fail.

Tested with Python 3.11.9 and the versions in `requirements-tested.txt`.

## Code

| File | Role |
|---|---|
| `code/lda/lda_bibliometric_analysis_v2.py` | Reading and deduplication of the BibTeX files, preprocessing, LDA, K selection, composite index, guideline dictionary and bibliographic tables |
| `code/lda/topic_correlations_and_map.py` | Document-topic correlations and intertopic distance map (pyLDAvis) of the K = 3 model |
| `code/lda/lda_sensitivity.py` | K and random seeds, composite-index weights, guidelines in the whole corpus |
| `code/lda/lda_guideline_stability.py` | Guideline shares in the digital transformation topic of the four LDA runs |
| `code/figures/make_figures.py` | Figures 1 to 12 of the manuscript, with English labels |

## Reading the results

Dictionary shares represent expressions located in titles and abstracts, not the proven presence or absence of a guideline in an article. The composite index is a descriptive ranking of the topics in this corpus, not a measure of scientific priority. The empirical interview and questionnaire materials remain in the controlled research archive and are not part of this public English-language package.

## License

Code: MIT (see `LICENSE`), as in the packages of the authors' other articles. See `NOTICE_THIRD_PARTY.md` for the material the license does not cover. Cite the article and this repository (`CITATION.cff`).
