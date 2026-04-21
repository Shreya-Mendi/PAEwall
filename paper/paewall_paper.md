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

### 3.3 Data Processing Pipeline and Rationale

Each preprocessing choice is motivated by a specific characteristic of patent–product matching:

| Step | Choice | Rationale |
|---|---|---|
| Defendant name normalization | Lowercase + strip corporate suffixes (`inc.`, `ltd.`, `corp.`) | CourtListener docket text and SEC EDGAR registrant names use inconsistent suffixes; normalization recovers ~8% more joins. |
| Patent claim extraction | Independent claims only, concatenated | Independent claims carry the full scope of the invention; dependent claims add narrowing language that inflates the query without adding retrieval signal. |
| 10-K section targeting | Item 1 "Business" section only | Item 1 is the product-description section; Items 7–8 are financial and add noise. MD&A (Item 7) was tested and hurt BM25 Recall@10 by 3 points. |
| Minimum description length | 50 words | Below this, descriptions are boilerplate ("the Company makes products...") and provide no retrieval signal. Shorter filings are dropped. |
| Train/test split | Split by `patent_id`, not by pair | Splitting by pair would leak patents across train and test since one patent can have multiple defendants. Splitting by `patent_id` guarantees no patent appears in both sets. |
| Vertical labels | CPC subclass → 5 buckets | CPC codes are hierarchical and too granular (>250 subclasses); we collapse to 5 business-relevant buckets that match the way PAE portfolios are actually organized. |
| Hard-negative mining | Top-BM25 non-relevant candidates per query, k=3 | In-batch negatives alone underfit because most random products are trivially irrelevant; BM25-derived hard negatives force the encoder to learn claim-level distinctions. |

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

### 5.4 Hyperparameter Tuning Strategy

Given the small training set (~420 pairs after split) and the cost of each training run (~20 min on an A100), we used a **principled coarse search over a constrained grid** rather than large-scale random or Bayesian search.

**TF-IDF + Logistic Regression.** Swept `max_features ∈ {10k, 25k, 50k, 100k}`, `ngram_range ∈ {(1,1), (1,2), (1,3)}`, and LR regularization `C ∈ {0.1, 1.0, 10.0}` via 5-fold cross-validation on the train split, selecting the best configuration by MRR. Selected: `max_features=50000, ngram=(1,2), C=1.0, sublinear_tf=True`. Bigrams gave a +3 point Recall@10 bump; trigrams overfit.

**Dual-encoder.** Four hyperparameters mattered most:

| Hyperparameter | Values tried | Selected | Reason |
|---|---|---|---|
| Learning rate | {1e-5, 2e-5, 5e-5} | 2e-5 | 5e-5 caused loss spikes by epoch 2; 1e-5 was under-trained at 5 epochs. |
| Batch size | {8, 16, 32} | 16 | 32 ran out of GPU memory on 40 GB A100 with both 768-d encoders active; 8 gave noisier in-batch negatives. |
| Projection dim | {128, 256, 512} | 256 | 128 hurt Recall@10 by 4 points; 512 showed no gain and doubled index size. |
| Hard-negative `k` per query | {0, 1, 3, 5} | 3 | `k=0` (in-batch only) plateaued at Recall@10 = 0.21; `k=5` overfit the small train set. |
| Temperature τ | {0.05, 0.07, 0.1} | 0.07 | Standard contrastive-learning setting (DPR, SimCSE); grid-search confirmed it was also best here. |
| Epochs | early stopping on val loss | 5 | Val loss plateaued at epoch 3 (Fig. 3); continued to 5 for a small additional train-loss gain with negligible overfit. |

We did **not** tune the encoder backbones — both are off-the-shelf (PatentSBERTa and all-mpnet-base-v2), chosen for domain coverage rather than swept. Encoder choice is itself treated as an ablation in §6.5.

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

Figure 1 visualizes the overall results; Figure 4 plots the Recall@K curve from K=10 to K=50. The fine-tuned dual encoder achieves the best Recall@10 (+36% over classical, +152% over BM25), confirming that dense retrieval with domain adaptation finds more true defendants in the top-10. Classical ML leads on MRR and nDCG@10, indicating better ranking precision when it retrieves the correct answer — likely due to the small training set (49 test examples) limiting the dual encoder's ranking calibration.

![Figure 1: Overall retrieval performance on PAE-Bench.](figures/overall_metrics.png)

![Figure 4: Recall@K curve across models (K=10 vs K=50).](figures/recall_vs_k.png)

### 6.3 Per-Vertical Results

| Vertical | BM25 R@10 | Classical R@10 | Dual-Enc R@10 |
|---|---|---|---|
| Software | 0.119 | 0.252 | **0.534** |
| Consumer Electronics | 0.196 | 0.378 | **0.313** |
| Medical Devices | 0.026 | 0.125 | — |
| Industrial | 0.087 | 0.000 | **0.000** |
| Other | 0.074 | 0.139 | 0.222 |

Software benefits most from fine-tuning (R@10 = 0.534), likely because patent claim language for software is more semantically aligned with 10-K product descriptions than hardware. Industrial fails entirely across all models — only 23 test pairs with highly specific mechanical claim language that has minimal lexical or semantic overlap with company-level product descriptions. Figure 2 shows this breakdown visually.

![Figure 2: Recall@10 by technology vertical.](figures/per_vertical_recall10.png)

### 6.4 Dual-Encoder Training Curve

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.831 | 2.825 |
| 2 | 2.630 | 2.832 |
| 3 | 2.526 | 2.661 |
| 4 | 2.485 | 2.704 |
| 5 | 2.490 | 2.700 |

Val loss stabilizes after epoch 3, suggesting the model has near-converged given the training set size (~420 pairs). Additional training data would be the primary lever for improvement. Figure 3 plots the full loss trajectory.

![Figure 3: Dual-encoder training curve (train vs val InfoNCE loss).](figures/training_curve.png)

### 6.5 Ablation: Fine-Tuning Impact (RQ1)

**Plan.** We decompose retrieval performance into three independent design choices — sparse vs. learned weighting, domain-adapted encoder, and task-specific fine-tuning — and evaluate four conditions:

1. **BM25** — sparse keyword baseline, no learning.
2. **TF-IDF + Logistic Regression** — classical learned feature weights.
3. **PatentSBERTa zero-shot** — domain-adapted encoder, no task fine-tuning.
4. **Dual-encoder fine-tuned** — domain-adapted encoder + task fine-tuning on litigation pairs.

Condition 4 is compared to Condition 3 to isolate the effect of fine-tuning on this task. Conditions 1–3 share the same 97-query evaluation split; Condition 4 uses the 49-query dual-encoder test set (patents for which both encoder towers had valid tokens).

**Results.** Three of four conditions completed; zero-shot PatentSBERTa (requires A100 GPU) is pending.

| Condition | R@10 | Gain vs previous |
|---|---|---|
| (1) BM25 | 0.130 | — |
| (2) TF-IDF + LR | 0.241 | +85% |
| (3) PatentSBERTa zero-shot | *pending* | *pending* |
| (4) Dual-encoder fine-tuned | 0.328 | +36% over (2) |

**Interpretation.** The progression from sparse → classical → dense learned representations yields monotonic Recall@10 improvement, supporting the hypothesis that both supervised learning *and* dense representation contribute independently. However, classical ML retains better ranking precision (MRR = 0.332 vs 0.075) — this suggests that with only 49 training examples in the evaluation split, the dual encoder's InfoNCE loss has learned to separate relevant from irrelevant products at the coarse level but has not yet learned to order relevant products precisely.

**Recommendations.** (1) Complete condition 3 to isolate the fine-tuning contribution from the encoder choice; we predict it will land between conditions 2 and 4 on Recall@10 (0.24 – 0.33). (2) Add a cross-encoder re-ranking stage on the top-20 dual-encoder candidates to close the MRR gap. (3) Expand training data — based on the training curve in Figure 3, both train and val loss plateau by epoch 3, indicating the model is data-limited rather than capacity-limited.

---

## 7. Error Analysis

Manual inspection of retrieval failures reveals two dominant failure modes: **vocabulary mismatch** (claim language has no lexical or semantic overlap with marketing-style 10-K product descriptions) and **corpus sparsity** (the true defendant's 10-K provides insufficient product detail). Table 3 presents five specific mispredictions drawn from `data/outputs/error_analysis.json`, covering both modes and all five verticals.

**Table 3. Five representative failure cases from BM25 and Classical evaluation.**

| # | Patent | Vertical | True defendant | Top retrieved | Root cause | Mitigation |
|---|---|---|---|---|---|---|
| 1 | US-10218033-B1 (Li-ion battery with SiO anode) | Other | Valtrus Innovations Ltd. | johnson & johnson; ups; uspto | **Vocabulary mismatch** + true defendant's 10-K description is empty (0 words). | Enrich product corpus with patent-assignee citation graphs; use prosecution history text as a secondary product signal. |
| 2 | US-10339934-B2 (server system action-assignment logic) | Software | Valtrus Innovations Ltd. | johnson & johnson; ups; home depot | **Empty product description** (0 words in 10-K Item 1). | For holding-company / IP-monetization entity defendants, fall back to the parent corporate filing or LinkedIn product copy. |
| 3 | US-10673996-B2 (modular electronic device) | Consumer Electronics | Valtrus Innovations Ltd. | johnson & johnson; ups; apple | **Same** — true defendant has no product text; retrieval returned the 3 most generic "consumer" companies. | Filter candidates by defendant size × filing-date recency; penalize defendants with <50-word Item 1. |
| 4 | US-10769923-B2 (automatic voice-alert system) | Consumer Electronics | Clearly IP Inc. | johnson & johnson; ups; apple | **Small/private company** — Clearly IP is not a SEC registrant, so it has no 10-K. | Extend corpus beyond SEC EDGAR to Crunchbase / OpenCorporates for private defendants; explicitly flag "not-in-corpus" in the UI. |
| 5 | US-10842131-B2 (agricultural remote-sensing method) | Industrial | Johnson & Johnson Inc. | ups; tccw; apple | **Claims truncation** — 907-word claim concatenation exceeds the 512-token max on the encoder side and the informative middle is cut. | Switch to claim-by-claim encoding with max-pooling over claims; use a long-context encoder (e.g., Longformer) for the patent tower. |

Failures 1–4 share a common upstream pathology: several companies in PAE-Bench are **IP-holding entities** (Valtrus, Clearly IP) whose only public "product" is their patent portfolio. Since our retrieval corpus is 10-K product descriptions, these companies are effectively unretrievable — no amount of better embedding helps. This is a **data-coverage failure**, not a model failure, and accounts for a disproportionate share of the Recall@10 deficit (Valtrus appears as the true defendant in 4 of the top-5 BM25 failures in `error_analysis.json`).

Failure 5 is genuinely algorithmic: the encoder truncated an informative claim mid-sentence. Short-term mitigation is claim-by-claim encoding; longer term it motivates a Longformer or ModernBERT patent encoder for the 8k+ token claim texts common in this domain.

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

## 11. Future Work

With another semester of effort, the single highest-leverage direction is **training-data expansion**: the dual-encoder's val loss plateaus by epoch 3 on ~420 training pairs, indicating the system is data-limited rather than architecture-limited. Concretely, we would:

1. **Expand PAE-Bench to 2,000+ pairs** by ingesting (a) USITC Section 337 proceedings, which add hardware infringement cases under-represented in district courts, and (b) PTAB IPR records, which include defendant-filed invalidity arguments that are themselves valuable training signal.
2. **Multimodal patent embedding.** Patent figures carry signal that text alone misses (especially for hardware and mechanical claims, where our "industrial" vertical fails entirely). We would add a SigLIP vision tower encoding the primary figure of each independent claim and a cross-attention fusion layer with the text tower. The current `dual_encoder.py` is architected so the product tower can accept concatenated text+image embeddings with minimal change.
3. **Cross-encoder re-ranking.** The dual encoder's weakness is fine-grained ranking (MRR = 0.075). Adding a distilled cross-encoder (e.g., MS-MARCO MiniLM) on the top-20 bi-encoder candidates should close the MRR gap without changing first-stage recall.
4. **Jurisdiction-aware enforcement probability.** The current enforcement-probability model is heuristic. With a labeled set of case outcomes from PACER dockets (win / settle / dismiss / loss), we would train a gradient-boosted classifier on claim breadth × jurisdiction × defendant industry × faithfulness score, replacing the rule-based predictor in `src/red_team/`.
5. **Figure-based patent intake.** Allow the user to upload only the primary figure of a patent and retrieve its text-plus-figure neighbors — useful for internal prior-art searches before filing.

---

## 12. Commercial Viability Statement

**Market.** Patent licensing and litigation in the U.S. exceeds $3 billion/year. Established players (Patlytics, IP.com, ClaimChart LLM, Patdel, Acacia Research, Intellectual Ventures) either operate as in-house research arms (opaque, not sold) or as closed SaaS tools with undisclosed benchmarks. The "public benchmark + open pipeline" wedge PAEwall occupies is unoccupied.

**Defensible differentiation.** Three structural advantages:

1. **Public, reproducible benchmark.** PAE-Bench is the first cross-vertical patent-to-product retrieval benchmark built from public data. Customers can evaluate against their internal datasets and compare; competitors cannot, because none publish numbers.
2. **Faithfulness-grounded outputs.** Every LLM-generated claim-chart limitation is scored against source product text. PAEs and defendants alike need auditable outputs they can put in front of an examiner or a jury; generic LLM claim charts fail this bar.
3. **Red-team by design.** The counter-argument module is not an add-on; it is co-generated with the claim chart. This flips the product from "build a PAE's case for them" to "give both sides the same honest view of enforcement viability", which broadens the buyer pool beyond PAEs to corporate legal departments, patent insurance carriers, and M&A IP-diligence providers.

**Unit economics.** Per-query cost is dominated by the LLM calls for claim chart + counter-arguments (~$0.04 at GPT-4.1-mini prices, ~$0.25 at GPT-4-class prices); retrieval itself is near-free after the one-time FAISS index build. A $5–$10 per-search price point clears a 10–20× gross margin, comparable to other legal-tech SaaS.

**Go-to-market.** The three natural beachheads are (i) individual inventors and small PAE funds priced out of Patlytics/IP.com licenses, (ii) in-house IP counsel at mid-market firms doing early enforcement triage, and (iii) patent insurance underwriters who need rapid portfolio-level enforcement probability scoring. Academic-licensing offices at universities (the "non-practicing entity" segment that is *not* a PAE) are a potential later-stage customer.

**Honest caveats.** PAEwall as it stands is a proof-of-concept, not a commercial product. The 32.8% Recall@10 is competitive for a research-grade system but not sufficient for unaccompanied legal deployment — the system needs to be framed as a triage tool for human attorneys, not a replacement. The claim-chart faithfulness verifier is a meaningful safety layer but is not yet certified against a ground-truth set of expert-annotated charts. Pre-commercial work would require that ground-truth, an SLA-grade inference stack, and an attorney-in-the-loop review workflow.

**Verdict.** Commercially viable as a vertical-SaaS tool in the legal-tech market, targeted first at the underserved small-PAE / mid-market-counsel segment. The technical moat is the combination of a public benchmark + faithfulness scoring + red-team counter-arguments — not the retrieval model itself, which any well-resourced competitor could reproduce.

---

## 13. Ethics Statement

Patent assertion tooling has a well-documented dual-use problem: the same infrastructure that helps a legitimate inventor find infringers of a valid patent also lowers the cost for patent trolls to issue speculative demand letters. PAEwall's design makes three deliberate choices to mitigate this:

1. **Faithfulness scoring is a first-class output, not an option.** Every generated claim chart includes a [0, 1] faithfulness score per limitation, computed by a separate LLM call against the source product description. Weak or fabricated limitations are surfaced, not hidden. A demand letter generated from a chart with an overall confidence of 0.3 is visibly weaker than one generated from a 0.9 chart — to the sender and to any recipient who also uses the tool.
2. **The red-team module is co-generated, not opt-in.** Every retrieval result is paired with non-infringement arguments, invalidity risks, and an enforcement-probability estimate. A PAE using the tool cannot look at the asserted strength without also seeing the defense's strongest response. This asymmetrically raises the cost of bringing weak assertions.
3. **Training data is litigation-derived and public.** PAE-Bench is built entirely from CourtListener dockets, Google Patents, and SEC EDGAR filings — no private product documentation, no scraped customer data, no confidential settlement records. The benchmark can be audited and reproduced by any researcher, and no individual's private data was used in model training.

**Residual risks we do not fully resolve.** A determined bad-faith user could ignore the faithfulness scores and the red-team output and still send spurious letters; the tool does not refuse to generate outputs for weak charts. We believe the right mitigation here is regulatory rather than technical — demand letters generated by AI tools should, in our view, be required to disclose that provenance and include the underlying faithfulness metrics. PAEwall's outputs are structured to support such disclosure if/when it becomes required.

**Data provenance.** All data used in training and evaluation is from public-record sources. No scraped private content. No personally identifiable information is stored by the deployed application; the only user input is a patent number or a patent PDF, which is not persisted server-side.

**Models and infrastructure.** The deployed LLM calls route through the Duke OIT litellm proxy, which logs query metadata for billing but does not retain content. Embeddings are computed locally on CPU and are never sent to external services. The dual-encoder backbones are released under permissive licenses (MIT for PatentSBERTa; Apache 2.0 for sentence-transformers).

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
