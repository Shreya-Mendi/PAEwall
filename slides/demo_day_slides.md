---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- PAEwall — Demo Day Slides
     Rubric deliverable:
       Problem & Motivation (1 slide) · Approach Overview (1 slide)
       Live Demo · Results / Insights (1 slide)
     Course: Duke MEng AIPI 540 · Deep Learning · Final Project
     5-min hard stop. Pairs with slides/demo_day_script.txt.
     Render: npx @marp-team/marp-cli@latest slides/demo_day_slides.md --pdf --allow-local-files
     Figures live in ../paper/figures/ relative to this file. -->

<style>
section { font-family: "Inter", -apple-system, sans-serif; background: #faf9f6; color: #1f2430; }
section.lead { background: linear-gradient(135deg, #eef2ff 0%, #faf9f6 60%, #d1fae5 100%); }
h1 { font-family: "Fraunces", Georgia, serif; letter-spacing: -0.5px; }
h2 { color: #4338ca; text-transform: uppercase; letter-spacing: 1.2px; font-size: 0.9rem; }
table { font-size: 0.78rem; }
th { background: #eef2ff; color: #3730a3; }
.highlight { background: linear-gradient(104deg, transparent 0%, #fde68a 5%, #fde68a 95%, transparent 100%); padding: 0 4px; }
.num { font-family: "Fraunces", Georgia, serif; color: #4338ca; font-weight: 700; }
code { background: #eef2ff; color: #3730a3; padding: 1px 6px; border-radius: 4px; }
</style>

<!-- _class: lead -->

# PAEwall
### Automated Patent Infringement Discovery

**Shreya Mendi** &nbsp;·&nbsp; Duke MEng AIPI 540 &nbsp;·&nbsp; Deep Learning Final Project &nbsp;·&nbsp; April 2026

<br>

> *Given a patent, find who's infringing it — and prove it.*

<br>

`github.com/Shreya-Mendi/PAEwall` &nbsp;·&nbsp; `paewall-production.up.railway.app`

---

## Problem & Motivation

# Patent trolls make <span class="num">$3B/year</span>. Search is still manual.

A patent attorney today:

**Reads claim language → guesses companies → hand-searches 10-Ks → builds a claim chart, line by line.**

<span class="highlight">$500+/hour · weeks per patent · the right defendant is often missed entirely.</span>

Tools that fix this — Patlytics, IP.com, ClaimChart LLM — cost **$50k+/year**. Individual inventors, small PAE funds, university tech-transfer, and in-house counsel are locked out.

### What's missing in the market

- A **public benchmark** — no one publishes numbers
- A **faithfulness score** on LLM-generated claim charts
- A **red-team module** that gives both sides the same honest view of enforcement viability

---

## Approach Overview

# Three models · one benchmark · four-module pipeline

**PAE-Bench** &nbsp;— 522 patent–defendant pairs from **real federal litigation (2015–2024)**, built from CourtListener + Google Patents BigQuery + SEC EDGAR 10-Ks. *First public cross-vertical patent-to-product retrieval benchmark.*

```
Patent → [ A. Intake ] → [ B. Retrieval ] → [ C. Faithfulness-Scored Chart ] → [ D. Red Team ]
```

| Model | Type | Recall@10 | Headline |
|---|---|---|---|
| BM25 | **Naive** | 13.0% | Keyword search — claim language ≠ marketing copy |
| TF-IDF + LogReg | **Classical ML** | 24.1% | Learned weights close part of the gap |
| **Fine-tuned Dual Encoder** | **Deep Learning** | **32.8%** | **+152% over BM25, +36% over classical** |

**Deep-learning recipe:** PatentSBERTa (patent tower, 3.1M patent pre-train) · all-mpnet-base-v2 (product tower) · shared `Linear(768→256)` projection · **InfoNCE contrastive loss** with k=3 BM25-mined hard negatives · FAISS flat-L2 index for sub-second inference.

---

<!-- _class: lead -->

## Live Demo

# 🎤 Live Demo · <span class="num">60 seconds</span>

**`paewall-production.up.railway.app`**

Patent: **`US-2014289857-A1`** — *Computer virus protection via email-attachment sandboxing*

### What you'll see the system do in real time

1. **Parse the patent** — independent claim auto-extracted into 6 labeled sub-parts (`1[a]` … `1[f]`)
2. **Retrieve** top candidate infringers — Microsoft, Palo Alto Networks, Symantec, Cisco, Proofpoint
3. **Highlight** — hover a claim-chart row → the matching sub-part of the patent lights up in yellow
4. **Score** — `[0,1]` faithfulness per limitation, verified by a separate LLM call against source product text
5. **Red team** — non-infringement arguments + §101/102/103 invalidity risks + enforcement probability

End-to-end: **< 10 seconds.** FastAPI · FAISS · Duke OIT LLM proxy.

---

## Results, Insights & Key Findings

![bg right:38% width:100%](../paper/figures/overall_metrics.png)

**Headline:** Fine-tuned dense retrieval is the right primitive for patent → product matching. <span class="highlight">**+152% Recall@10 over BM25.**</span>

**Software vertical wins biggest:** R@10 = **53.4%** with the dual encoder — semantic match works when both sides are text-rich.

**Classical still wins MRR:** ranking calibration is data-limited at n=49 test. **Cross-encoder re-ranker is the next lever.**

**Error analysis** surfaced a data-coverage issue, not a model issue: **4 of top-5 BM25 misses are IP-holding entities** (Valtrus, Clearly IP) with empty 10-K product sections. Benchmark expansion (USITC §337 + PTAB IPR) is the next semester's highest-leverage move.

**Training curve plateaus at epoch 3** → the model is hungry for data, not capacity. Confirms the "expand the benchmark" recommendation.

**Commercial wedge:** public benchmark + faithfulness scoring + co-generated red-team output is **unoccupied** in the legal-tech market. **Ethics is built into the product**, not bolted on — every claim chart ships with a faithfulness score; every retrieval ships with the defense's strongest argument.

---

<!-- _class: lead -->

# Thank you

**Shreya Mendi** · `shreya.mendi@duke.edu`

Code: `github.com/Shreya-Mendi/PAEwall`
Live: `paewall-production.up.railway.app`
Paper: `paper/paewall_paper.pdf` · `.md` · `.tex`
