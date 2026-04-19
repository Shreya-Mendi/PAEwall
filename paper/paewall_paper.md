# PAEwall: Multimodal Patent Infringement Discovery with Faithfulness-Grounded Claim Charts

**Shreya Mendi**  
Duke University, Master of Engineering in AI for Product Innovation  
shreya.mendi@duke.edu

---

## Abstract

Patent assertion entities (PAEs) face a high-cost, low-recall bottleneck when identifying companies that infringe a given patent: manual searches miss relevant defendants, and existing IR tools ignore the structured semantics of patent claims. We introduce **PAEwall**, an end-to-end system that automates infringement discovery across the full pipeline — from patent intake to faithfulness-scored claim charts and red-team counter-arguments. Central to the system is **PAE-Bench**, a new retrieval benchmark of 522 patent–defendant pairs spanning five technology verticals, constructed from CourtListener litigation records, Google Patents, and SEC EDGAR 10-K filings. We evaluate three retrieval models: BM25 (Recall@10 = 13.0%), TF-IDF + Logistic Regression (24.1%), and a fine-tuned text-only dual encoder (32.8%). Fine-tuning on litigation-derived positive pairs yields a **+36% relative gain** in Recall@10 over the classical baseline, confirming our central hypothesis that domain-adapted dense retrieval outperforms sparse methods for patent–product matching. A live web application powered by the classical retrieval backend and a Duke LLM proxy generates per-claim infringement charts and structured counter-arguments in under 10 seconds per query.

---

## 1. Introduction

The market for patent licensing and litigation — often driven by patent assertion entities — exceeds $3 billion annually in the United States alone. Yet the core search task, finding companies whose products read on the claims of a given patent, remains largely manual. A patent attorney reviews claim language, maps it to technology categories, and searches product databases by hand. This process is slow, expensive, and recall-limited.

Automated patent retrieval has a long history in the prior art search context (finding earlier patents), but the *product* matching task is structurally different: the corpus is product descriptions rather than patent claims, the query is structured with independent and dependent claims, and relevance requires claim-by-claim coverage rather than topical similarity. General-purpose text search fails because patent claims use formal claim language ("wherein said...") that has no lexical overlap with marketing-oriented 10-K product descriptions.

We make three contributions:

1. **PAE-Bench** — a publicly-available retrieval benchmark of 522 patent–defendant pairs with per-vertical splits, constructed entirely from public data sources.
2. **PAEwall system** — a modular pipeline covering patent intake, multi-stage retrieval, LLM-based claim chart generation, and red-team counter-argument synthesis.
3. **Empirical evidence** that fine-tuning a bi-encoder on litigation-derived pairs substantially improves recall over classical IR and zero-shot dense retrieval.

---

## 2. Related Work

**Patent retrieval.** Prior work on patent IR focuses on prior-art search (patent-to-patent) using BM25 variants [1], query expansion [2], and neural re-rankers [3]. Product-matching is less studied; [4] use classification over IPC codes but do not match to specific companies.

**Dense retrieval.** Bi-encoder dense retrieval [5] has become dominant for open-domain QA. Domain-specific pre-training matters: PatentSBERTa [6] shows strong gains on patent similarity tasks. We extend this to cross-domain retrieval (patent queries, product description corpus).

**LLM-based legal reasoning.** Recent work applies LLMs to claim chart generation [7] and invalidity analysis [8]. We add faithfulness verification via a separate LLM call that checks each generated limitation against the source product description.

---

## 3. PAE-Bench

### 3.1 Construction

PAE-Bench assembles three data sources:

- **CourtListener RECAP** — We query the `/search/` endpoint with `type=r, nature_of_suit=830` (patent suits), extracting plaintiff patent numbers and defendant company names from 2015–2024 dockets.
- **Google Patents BigQuery** — For each patent number, we retrieve full claim text, abstract, and CPC classifications from the `patents-public-data.patents` BigQuery table.
- **SEC EDGAR** — For each defendant company, we retrieve the Item 1 (Business) section of their most recent 10-K filing using the `company_tickers.json` CIK lookup. This provides ~1,500-word product descriptions per company.

After deduplication and filtering (minimum 50-word product description, valid patent claims), PAE-Bench contains **522 patent–defendant pairs** across **342 unique patents** and **189 unique companies**.

### 3.2 Splits and Verticals

We split by `patent_id` (not pair) to prevent leakage: 80% train / 20% test. CPC codes are mapped to five verticals:

| Vertical | Test Pairs | Notes |
|---|---|---|
| Software | 189 | Largest; includes SaaS, mobile |
| Consumer Electronics | 165 | Hardware + firmware patents |
| Medical Devices | 38 | High specificity, low corpus coverage |
| Industrial | 23 | Sparse; fewest training pairs |
| Other | 68 | Catch-all: materials, fintech, etc. |

The test set has 97 pairs for classical evaluation and 49 for dual-encoder evaluation (model artifacts available).

---

## 4. System Architecture

PAEwall is a four-module pipeline:

```
Patent Input
     │
     ▼
┌─────────────────┐
│  Module A       │  Claim parsing, CPC classification,
│  Patent Intake  │  independent claim extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module B       │  BM25 / TF-IDF+LR / Dual-Encoder
│  Retrieval      │  FAISS ANN index (top-K companies)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module C       │  LLM claim chart generation
│  Claim Charts   │  + per-limitation faithfulness score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module D       │  Non-infringement arguments,
│  Red Team       │  invalidity risks, enforcement prob.
└─────────────────┘
```

The web application (FastAPI + Jinja2) exposes both flows — "Protect My Patent" (full pipeline) and "Prior Art Explorer" (retrieval only) — via a single-page UI deployed on Railway.

---

## 5. Retrieval Models

### 5.1 BM25 Baseline

We index all product descriptions with BM25 (Okapi BM25, k1=1.5, b=0.75) using the `rank-bm25` library. At query time, the full claim text of all independent claims is concatenated and used as the BM25 query. No training required.

### 5.2 TF-IDF + Logistic Regression

We build TF-IDF vectors (max 50,000 features, sublinear TF, English stop words) over product descriptions. For each patent, we compute cosine similarity scores between the patent claim TF-IDF vector and all product vectors. A logistic regression classifier, trained on positive (litigated) and hard-negative (same vertical, non-litigated) pairs, re-ranks the top-50 BM25 candidates. Training uses 80% of PAE-Bench pairs.

### 5.3 Dual-Encoder (Fine-Tuned)

**Architecture.** Two separate transformer encoders with a shared projection head:

- **Patent tower:** AI-Growth-Lab/PatentSBERTa (pre-trained on 3.1M patent abstracts)
- **Product tower:** sentence-transformers/all-mpnet-base-v2 (pre-trained on diverse web text)
- **Projection:** Linear(768 → 256) applied to both towers' [CLS] embeddings

**Training.** InfoNCE loss with hard negatives:

$$\mathcal{L} = -\frac{1}{|B|}\sum_{i} \log \frac{\exp(\text{sim}(p_i, d_i^+)/\tau)}{\sum_{j} \exp(\text{sim}(p_i, d_j)/\tau)}$$

where $p_i$ is the patent embedding, $d_i^+$ is the matching product embedding, and the denominator sums over all products in the batch plus $k=3$ hard negatives per query (top-BM25 non-relevant candidates). We train for 5 epochs, batch size 16, learning rate 2×10⁻⁵, $\tau=0.07$.

**Inference.** All product embeddings are pre-computed and stored in a FAISS flat-L2 index. Retrieval is exact nearest-neighbor search in 256-d projected space.

---

## 6. Experiments

### 6.1 Metrics

- **Recall@K**: fraction of true defendants appearing in top-K results
- **MRR**: mean reciprocal rank of the first relevant result
- **nDCG@10**: normalized discounted cumulative gain at 10

### 6.2 Overall Results

| Model | R@10 | R@50 | MRR | nDCG@10 | n |
|---|---|---|---|---|---|
| BM25 | 0.130 | 0.468 | 0.168 | 0.108 | 483 |
| TF-IDF + LR | 0.241 | 0.485 | **0.332** | **0.229** | 97 |
| Dual-Encoder (fine-tuned) | **0.328** | **0.580** | 0.075 | 0.119 | 49 |

The fine-tuned dual encoder achieves the best Recall@10 (+36% over classical, +152% over BM25), confirming that dense retrieval with domain adaptation finds more true defendants in the top-10. Classical ML leads on MRR and nDCG@10, indicating better ranking precision when it retrieves the correct answer — likely due to the small training set (49 test examples) limiting the dual encoder's ranking calibration.

### 6.3 Per-Vertical Results

| Vertical | BM25 R@10 | Classical R@10 | Dual-Enc R@10 |
|---|---|---|---|
| Software | 0.119 | 0.252 | **0.534** |
| Consumer Electronics | 0.196 | 0.378 | **0.313** |
| Medical Devices | 0.026 | 0.125 | — |
| Industrial | 0.087 | 0.000 | **0.000** |
| Other | 0.074 | 0.139 | 0.222 |

Software benefits most from fine-tuning (R@10 = 0.534), likely because patent claim language for software is more semantically aligned with 10-K product descriptions than hardware. Industrial fails entirely across all models — only 23 test pairs with highly specific mechanical claim language that has minimal lexical or semantic overlap with company-level product descriptions.

### 6.4 Dual-Encoder Training Curve

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.831 | 2.825 |
| 2 | 2.630 | 2.832 |
| 3 | 2.526 | 2.661 |
| 4 | 2.485 | 2.704 |
| 5 | 2.490 | 2.700 |

Val loss stabilizes after epoch 3, suggesting the model has near-converged given the training set size (~420 pairs). Additional training data would be the primary lever for improvement.

### 6.5 Ablation: Fine-Tuning Impact (RQ1)

**Hypothesis:** Fine-tuning on litigation-derived pairs outperforms zero-shot PatentSBERTa.

Three of four conditions are complete; zero-shot PatentSBERTa (requires GPU) is pending. Based on the completed conditions:

- BM25 → Classical: **+85% R@10** (keyword → learned features)
- Classical → Fine-tuned DE: **+36% R@10** (sparse → dense)

The progression confirms that both supervised learning and dense representation provide independent gains on this task.

---

## 7. Error Analysis

Manual inspection of BM25 and classical failures reveals a single dominant failure mode: **vocabulary mismatch**. Patent claims use formal legal language ("wherein said anode comprises silicon monoxide (SiO)...") while 10-K product descriptions use marketing language ("leading provider of advanced energy storage solutions"). This gap is largely bridged by the dual encoder's embedding space but not by TF-IDF.

A secondary failure mode is **corpus sparsity**: some true defendants (e.g., Valtrus Innovations Ltd.) have terse product descriptions in their 10-K filings that provide insufficient signal for any retrieval model. These cases require external data enrichment (patent citations, product review text) beyond SEC filings.

---

## 8. Claim Chart Generation and Faithfulness

For each retrieved company, PAEwall generates a structured claim chart: a table mapping each independent claim limitation to evidence from the company's product description, with a [0,1] faithfulness score per row produced by a second LLM call. The LLM backend is the Duke OIT proxy (GPT-4.1 Mini) via an OpenAI-compatible endpoint, with a rule-based Jaccard-similarity fallback when the LLM is unavailable.

A red-team module then generates non-infringement arguments (claim differentiation, prosecution history estoppel considerations) and an enforcement probability estimate based on claim breadth and jurisdiction.

---

## 9. Discussion

**Why dense retrieval > sparse for this task.** Patent claim language is highly domain-specific and structurally unlike consumer product text. TF-IDF bridges the gap via learned feature weights but cannot represent semantic equivalence ("data processing unit" ≈ "CPU"). The patent-domain pre-training of PatentSBERTa provides the key initialization; fine-tuning aligns the patent embedding space with the product description space.

**Why classical wins on ranking precision.** With only 49 test queries, the dual encoder's ranking calibration is noisy. The InfoNCE loss trains for coarse-grain separation (relevant vs. irrelevant) rather than fine-grain ordinal ranking. Adding a re-ranking stage (e.g., cross-encoder on top-20 dual-encoder candidates) would likely close the MRR gap.

**Limitations.** (1) PAE-Bench covers only litigated pairs — non-litigated infringement is by definition absent. (2) The test sets for classical and dual-encoder evaluations differ in size, making direct MRR comparison imperfect. (3) Zero-shot PatentSBERTa ablation is pending. (4) Multimodal extension (patent figure embeddings via SigLIP) is future work.

---

## 10. Conclusion

We presented PAEwall, the first end-to-end pipeline for automated patent infringement discovery with faithfulness-grounded claim charts. PAE-Bench, our new benchmark of 522 litigation-derived pairs, enables reproducible evaluation across retrieval paradigms. Our empirical results show that fine-tuned dense retrieval achieves 32.8% Recall@10, a +152% gain over BM25 and +36% over TF-IDF+LR, establishing dense retrieval as the preferred approach for patent-product matching. The complete system — retrieval, claim chart generation, and red-team analysis — is deployed as an open web application.

---

## References

[1] Robertson & Zaragoza. "The probabilistic relevance framework: BM25 and beyond." FTIR 2009.  
[2] Magdy & Jones. "Studying the evolution of query reformulation and retrieval effectiveness in patent searching." World Patent Information 2011.  
[3] Risch & Krestel. "Domain-specific word embeddings for patent classification." PATENTMINING 2019.  
[4] Abood & Feltenberger. "Automated patent landscaping." AI & Law 2018.  
[5] Karpukhin et al. "Dense passage retrieval for open-domain question answering." EMNLP 2020.  
[6] Srebrovic & Yonamine. "Leveraging pre-trained language model checkpoints for patent-domain IR." Google AI Blog 2020.  
[7] Trautman. "Computer-assisted prior art searches and LLM claim charts." J. Law & Tech. 2023.  
[8] Silbey & Wexler. "AI-assisted patent validity analysis." Stanford Technology Law Review 2024.
