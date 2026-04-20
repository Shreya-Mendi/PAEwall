---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- PAEwall — Demo Day Slides
     5-minute hard stop. Render with: marp slides/demo_day_slides.md --pdf
     Figures live in ../paper/figures/ relative to this file. -->

<!-- _class: lead -->

# PAEwall
### Automated Patent Infringement Discovery

**Shreya Mendi** · Duke MEng AIPI · April 2026

> *Given a patent, find who's infringing it — and prove it.*

`github.com/Shreya-Mendi/PAEwall`

---

## The Problem

> **U.S. patent licensing & litigation is a $3B+/year market.**
> The core search task is still done by hand.

A patent attorney today does this manually:

1. Read the claims.
2. Guess which companies might infringe.
3. Search product pages, 10-Ks, marketing copy.
4. Build a claim chart, line by line.

**Cost:** $500+/hour · weeks per patent · low recall (the right defendant is often missed)

**Who loses:** Individual inventors, small PAE funds, university tech-transfer offices, and corporate counsel doing early enforcement triage — anyone priced out of $50k+/year tools like Patlytics or IP.com.

---

## Our Approach: Three Models, One Benchmark

**PAE-Bench** — 522 patent–defendant pairs from real US litigation (2015–2024), built from CourtListener + Google Patents BigQuery + SEC EDGAR 10-Ks. *First public cross-vertical benchmark of its kind.*

```
Patent in → [Intake] → [Retrieval] → [Claim Chart + Faithfulness] → [Red Team]
```

We trained three retrievers and benchmarked them head-to-head:

| Model | Recall@10 | What it shows |
|---|---|---|
| BM25 (naive) | 13.0% | Keyword search alone fails — claim language ≠ marketing copy |
| TF-IDF + LR (classical) | 24.1% | Learned weights help, but no semantic understanding |
| **Dual-Encoder (fine-tuned)** | **32.8%** | **+152% over BM25, +36% over classical** |

Patent tower: PatentSBERTa · Product tower: all-mpnet-base-v2 · InfoNCE loss + hard negatives

---

## Live Demo

# 🎤 Live Demo

**`<your-railway-url>.up.railway.app`**

What you'll see in 60 seconds:

1. Paste a real US patent number → claims auto-parsed.
2. Top-10 candidate infringers ranked, with vertical labels.
3. **Faithfulness-scored claim chart** — every limitation mapped to product evidence with a [0,1] confidence score.
4. **Red-team output** — non-infringement arguments + invalidity risks + enforcement probability.

End-to-end inference: **< 10 seconds.** FastAPI + FAISS + Duke LLM proxy.

---

## Results & Insights

![bg right:42% width:95%](../paper/figures/overall_metrics.png)

**Headline:** Fine-tuned dense retrieval is the right primitive for patent → product matching.

**Software vertical wins biggest:** R@10 = 53% with the dual encoder — semantic match works when both sides are text-rich.

**Industrial vertical fails everywhere:** highly specialized mechanical claim language has no overlap with company-level descriptions. Future work needs patent-figure embeddings (SigLIP).

**Classical wins on MRR/nDCG:** dual encoder finds more answers; classical ranks them more precisely. → cross-encoder re-ranker is the next lever.

**Failure analysis** surfaced an upstream pathology: 4 of top-5 BM25 misses are IP-holding entities (Valtrus, Clearly IP) with empty 10-K product sections. **Data-coverage problem**, not a model problem.

---

## What's Next + The Commercial Wedge

**Future work** (unchanged-architecture, biggest leverage first):

- **Expand PAE-Bench → 2,000+ pairs** via USITC §337 + PTAB IPR records
- **Multimodal** — SigLIP patent-figure tower feeding the product encoder
- **Cross-encoder re-rank** on top-20 candidates → close MRR gap
- **Jurisdiction-aware enforcement model** trained on PACER outcomes

**Why this is fundable:**

- Public benchmark + faithfulness scoring + red-team module is **unoccupied** in the market — Patlytics, IP.com, ClaimChart LLM publish nothing.
- Per-query cost ~$0.04 (LLM-bound) · clean $5–$10 SaaS price point · 10–20× gross margin.
- **Honest framing:** triage tool for human attorneys, not a replacement.

**Ethics built-in:** every claim chart ships with a faithfulness score; every retrieval result ships with the strongest defense argument. The tool *raises* the cost of frivolous assertion.

---

<!-- _class: lead -->

# Thank you

**Shreya Mendi** · `shreya.mendi@duke.edu`

`github.com/Shreya-Mendi/PAEwall`

*Live demo: `<your-railway-url>.up.railway.app`*
*Paper: `paper/paewall_paper.md` · `paper/paewall_paper.tex`*
