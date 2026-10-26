# Rezultate grila RAG

128 configurații × 4 strategii de chunking. Generator: extractiv (offline). Cost: estimat pentru Claude Sonnet 5.5 ($2 / $10 per 1M tokeni).

## Top 10 după answer recall

| configurație | hit@k | MRR | ctx prec. | ctx recall | answer recall | answer F1 | refuz corect | refuz greșit | tokeni ctx | $ / 1k întrebări | latență ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| section|hybrid|no-rerank|5 | 0.983 | 0.880 | 0.200 | 0.983 | 0.644 | 0.325 | 0.200 | 0.117 | 450 | 1.214 | 0.28 |
| section|char-ngram|no-rerank|5 | 0.967 | 0.866 | 0.197 | 0.967 | 0.644 | 0.331 | 0.200 | 0.117 | 465 | 1.237 | 0.16 |
| section|bm25|no-rerank|8 | 1.000 | 0.848 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 646 | 1.606 | 0.04 |
| section|bm25|rerank|8 | 1.000 | 0.836 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 662 | 1.637 | 0.89 |
| section|tfidf|rerank|8 | 1.000 | 0.835 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 662 | 1.639 | 0.98 |
| section|hybrid|no-rerank|8 | 1.000 | 0.882 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 668 | 1.650 | 0.28 |
| section|hybrid|rerank|8 | 1.000 | 0.838 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 674 | 1.662 | 1.41 |
| section|char-ngram|rerank|8 | 1.000 | 0.839 | 0.127 | 1.000 | 0.636 | 0.322 | 0.200 | 0.117 | 695 | 1.703 | 1.29 |
| section|bm25|no-rerank|5 | 0.983 | 0.846 | 0.200 | 0.983 | 0.636 | 0.322 | 0.200 | 0.117 | 434 | 1.182 | 0.04 |
| section|tfidf|no-rerank|5 | 0.983 | 0.854 | 0.200 | 0.983 | 0.636 | 0.322 | 0.200 | 0.117 | 434 | 1.183 | 0.04 |

## Frontiera Pareto (calitate vs. cost)

| configurație | hit@k | MRR | ctx prec. | ctx recall | answer recall | answer F1 | refuz corect | refuz greșit | tokeni ctx | $ / 1k întrebări | latență ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-50|hybrid|rerank|1 | 0.767 | 0.767 | 0.833 | 0.756 | 0.548 | 0.320 | 0.400 | 0.133 | 150 | 0.522 | 1.56 |
| section|tfidf|no-rerank|1 | 0.767 | 0.767 | 0.767 | 0.760 | 0.589 | 0.335 | 0.200 | 0.150 | 152 | 0.552 | 0.04 |
| section|bm25|no-rerank|1 | 0.750 | 0.750 | 0.750 | 0.744 | 0.593 | 0.337 | 0.200 | 0.133 | 153 | 0.557 | 0.04 |
| section|char-ngram|no-rerank|1 | 0.800 | 0.800 | 0.800 | 0.794 | 0.597 | 0.336 | 0.200 | 0.167 | 153 | 0.557 | 0.16 |
| section|hybrid|no-rerank|1 | 0.800 | 0.800 | 0.800 | 0.794 | 0.616 | 0.352 | 0.200 | 0.133 | 153 | 0.559 | 0.28 |
| section|tfidf|no-rerank|3 | 0.967 | 0.850 | 0.328 | 0.967 | 0.630 | 0.324 | 0.200 | 0.117 | 293 | 0.891 | 0.04 |
| section|bm25|rerank|3 | 0.933 | 0.822 | 0.317 | 0.933 | 0.631 | 0.321 | 0.200 | 0.117 | 302 | 0.914 | 0.89 |
| section|hybrid|no-rerank|3 | 0.950 | 0.872 | 0.322 | 0.950 | 0.635 | 0.324 | 0.200 | 0.117 | 304 | 0.918 | 0.28 |
| section|bm25|no-rerank|5 | 0.983 | 0.846 | 0.200 | 0.983 | 0.636 | 0.322 | 0.200 | 0.117 | 434 | 1.182 | 0.04 |
| section|hybrid|no-rerank|5 | 0.983 | 0.880 | 0.200 | 0.983 | 0.644 | 0.325 | 0.200 | 0.117 | 450 | 1.214 | 0.28 |

## Efectul dimensiunii `chunking` (medie pe restul grilei)

| chunking | hit@k | MRR | ctx recall | answer recall | tokeni ctx |
|---|---:|---:|---:|---:|---:|
| section | 0.920 | 0.825 | 0.918 | 0.625 | 394 |
| fixed-50 | 0.914 | 0.828 | 0.911 | 0.571 | 401 |
| fixed-100 | 0.935 | 0.852 | 0.932 | 0.614 | 692 |
| fixed-200 | 0.946 | 0.869 | 0.943 | 0.597 | 1194 |

## Efectul dimensiunii `retriever` (medie pe restul grilei)

| retriever | hit@k | MRR | ctx recall | answer recall | tokeni ctx |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.923 | 0.830 | 0.921 | 0.600 | 662 |
| tfidf | 0.923 | 0.831 | 0.921 | 0.602 | 657 |
| char-ngram | 0.934 | 0.856 | 0.932 | 0.601 | 683 |
| hybrid | 0.934 | 0.857 | 0.931 | 0.603 | 678 |

## Efectul dimensiunii `rerank` (medie pe restul grilei)

| rerank | hit@k | MRR | ctx recall | answer recall | tokeni ctx |
|---|---:|---:|---:|---:|---:|
| False | 0.939 | 0.861 | 0.936 | 0.603 | 666 |
| True | 0.919 | 0.826 | 0.916 | 0.601 | 675 |

## Efectul dimensiunii `k` (medie pe restul grilei)

| k | hit@k | MRR | ctx recall | answer recall | tokeni ctx |
|---|---:|---:|---:|---:|---:|
| 1 | 0.784 | 0.784 | 0.776 | 0.569 | 220 |
| 3 | 0.947 | 0.857 | 0.946 | 0.609 | 496 |
| 5 | 0.987 | 0.866 | 0.986 | 0.614 | 778 |
| 8 | 0.996 | 0.867 | 0.996 | 0.614 | 1187 |

## hit@3 pe tipuri de întrebări (chunking pe secțiuni, fără reranking)

| retriever | contrast | direct | multi | paraphrase |
|---|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 1.000 | 0.875 |
| tfidf | 1.000 | 0.973 | 1.000 | 0.938 |
| char-ngram | 1.000 | 0.946 | 1.000 | 0.812 |
| hybrid | 1.000 | 0.973 | 1.000 | 0.875 |

## Pragul de refuz al generatorului (`section|hybrid|no-rerank|5`)

| prag | refuz corect (fără răspuns) | refuz greșit (cu răspuns) | answer recall |
|---:|---:|---:|---:|
| 0.30 | 0.20 | 0.033 | 0.681 |
| 0.40 | 0.20 | 0.117 | 0.644 |
| 0.50 | 0.40 | 0.217 | 0.592 |
| 0.55 | 0.40 | 0.300 | 0.546 |
| 0.60 | 0.40 | 0.300 | 0.546 |
| 0.70 | 0.60 | 0.367 | 0.503 |
| 0.80 | 0.60 | 0.467 | 0.434 |
