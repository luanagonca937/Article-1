# Third-party material

The MIT license in `LICENSE` covers the code written by the authors (`reproduce_article1.py`, the run scripts and `code/`). It does not cover the material below.

## Included

| File | Content |
|---|---|
| `data/unique_records_1272.csv`, `data/modelled_records_1249.csv` | Bibliographic metadata of the retrieved records (title, year, source, DOI, document type and dominant topic), without abstracts. The metadata keep the terms of use of the databases and publishers and are included only so that the corpus can be identified. |

## Not included

- The BibTeX exports of ScienceDirect, Scopus, Web of Science and IEEE Xplore, and any file derived from them with abstracts (subject to the license terms of the databases). Export them again with `data/SEARCH_PROTOCOL.md`; `.gitignore` excludes `data/raw_bib/*.bib`.
- Passages of published abstracts matched by the guideline dictionary (the `Context` column was removed from `archived_results/13_guideline_stability.xlsx`).
- Full texts of the cited articles (copyrighted).

If you hold rights to any material here and want it changed or removed, please open an issue.
