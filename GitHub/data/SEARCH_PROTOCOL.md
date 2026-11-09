# Search protocol

Search string (the same in every database, adapted only to the syntax of each interface):

```
("implementation guideline" OR "Hoshin Kanri" OR "strategy deployment") AND ("digital transformation" OR "smart farming" OR "livestock" OR "animal production" OR "agribusiness" OR "cooperative")
```

| Database | Date of the search | Records retrieved | Filter applied before export | Records exported (BibTeX) |
|---|---|---:|---|---:|
| ScienceDirect (Advanced search) | 1 November 2026 | 1,778 | Research articles | 1,256 (13 files) |
| Scopus | 1 November 2026 | 57 | Article | 28 |
| Web of Science | 1 November 2026 | 5 | Article | 2 |
| IEEE Xplore | 1 November 2026 (record downloaded again with its abstract on 2 November 2026) | 9 | Journals | 1 |
| **Total** | | **1,849** | | **1,287** |

No restriction on publication year was applied. Export the records as BibTeX with abstracts and keywords and put the 16 files in `raw_bib/`.

Fields searched in each database: [authors to confirm].

After deduplication by normalised title the corpus has 1,272 unique records (`unique_records_1272.csv`); the 1,249 records with an abstract of at least 100 characters were modelled (`modelled_records_1249.csv`, with the dominant topic of each record). Because the databases add records over time, a new export can differ from the one used in the article, and some values may then fail the verification.
