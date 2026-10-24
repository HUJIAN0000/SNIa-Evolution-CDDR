# Data, source attribution and license scope

`LICENSE` is copied unchanged from the existing SNIa-Evolution-CDDR repository, copyright 2026 HUJIAN0000, MIT license. It preserves the existing analysis-project license rather than inventing a new copyright holder or software-contributor list.

The following components retain their own provenance and rights. The root analysis license does not relicense third-party catalogs or a separately developed vendor component.

| Component | Origin | Notes |
|---|---|---|
| Pantheon+ catalog and STAT+SYS covariance | Author-supplied snapshot of [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease) | Cite Brout et al. 2022 and Scolnic et al. 2022; original scientific data rights remain |
| 120 radio sources | Cao et al. 2017, A&A 606 A15, DOI 10.1051/0004-6361/201730551 | Supplied CSV is preserved; no article PDF is redistributed |
| `code/vendor/CosmoMatcherv1.4.3.py` | Exact author-supplied CosmoMatcher backend, [upstream](https://github.com/HUJIAN0000/CosmoMatcher) | The upstream licensing-status statement is preserved in `code/vendor/LICENSING_STATUS.upstream.md`; this package does not assign MIT or another license to that separate component |
| `original/` analysis and plotting sources | Effective first-draft SNIa-Evolution-CDDR scripts | Retained for historical matching and likelihood comparison |
| Python dependencies | NumPy, SciPy/HiGHS, pandas, emcee, GetDist, Matplotlib, corner, PyMuPDF, Pillow | Keep their upstream licenses; installed through requirements rather than vendored |

The frozen generated figures and result arrays are associated with this analysis and require appropriate scientific attribution. No AASTeX class, referee correspondence, manuscript PDF or third-party reference-paper PDF is included. Software citation metadata and any DOI deposit should use confirmed actual software contributors; the six-person paper author list is not automatically a software-author list.
