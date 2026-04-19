# PAEwall — Demo Day Slides
<!-- 5 slides · Marp-compatible markdown (use Marp CLI or VS Code Marp extension to render) -->
<!-- marp: true, theme: default, paginate: true -->

---

## Slide 1 — Title

# PAEwall
### Automated Patent Infringement Discovery

**Shreya Mendi** · Duke MEng AIPI · April 2026

> *Given a patent, find who is infringing it — and prove it.*

**Problem:** Patent holders spend weeks manually identifying potential infringers.  
Manual search misses defendants. Legal review is $500+/hour.

**PAEwall automates the full pipeline:**  
Patent in → Defendants ranked → Claim charts generated → Counter-arguments ready

---

## Slide 2 — Dataset: PAE-Bench

# PAE-Bench: A New Retrieval Benchmark

**522 patent–defendant pairs** from real US litigation (2015–2024)

| Source | What We Collected |
|---|---|
| CourtListener RECAP | Patent numbers + defendant companies from suit filings |
| Google Patents BigQuery | Full claim text + CPC classifications |
| SEC EDGAR 10-K | Item 1 product descriptions per defendant company |

**5 verticals:** Software · Consumer Electronics · Medical Devices · Industrial · Other

**Why it's hard:** Patent claims use legal language ("wherein said processor...").  
Product descriptions use marketing language ("industry-leading solution...").  
No lexical overlap → keyword search fails.

---

## Slide 3 — Three Models, One Benchmark

# Retrieval Results on PAE-Bench

| Model | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 (keyword baseline) | 13.0% | 0.168 | 0.108 |
| TF-IDF + LogReg | 24.1% | **0.332** | **0.229** |
| **Dual-Encoder (fine-tuned)** | **32.8%** | 0.075 | 0.119 |

**+152% Recall@10 gain** from BM25 → fine-tuned dense retrieval

**How the dual encoder works:**
- Patent tower: PatentSBERTa (pre-trained on 3.1M patent abstracts)
- Product tower: all-mpnet-base-v2
- Trained with InfoNCE loss + hard negatives on litigation pairs
- FAISS index for sub-second inference at 189 companies

**Key finding:** Dense retrieval finds more defendants (+36% vs. classical).  
Classical is more precise when it finds them (higher MRR/nDCG).

---

## Slide 4 — The Full Pipeline

# PAEwall System: 4 Modules

```
Patent Number / PDF
        │
  ┌─────▼──────┐   Claim parsing, CPC classification
  │  Intake    │   → extract independent claims
  └─────┬──────┘
        │
  ┌─────▼──────┐   TF-IDF+LR retrieval (live)
  │ Retrieval  │   Dual-encoder + FAISS (GPU, fallback)
  └─────┬──────┘   → ranked list of companies
        │
  ┌─────▼──────┐   GPT-4.1 Mini via Duke LLM proxy
  │Claim Charts│   → per-limitation evidence mapping
  │            │   → faithfulness score [0–1] per row
  └─────┬──────┘
        │
  ┌─────▼──────┐   Non-infringement arguments
  │  Red Team  │   Invalidity risks
  └────────────┘   Enforcement probability estimate
```

**Live demo:** [paewall.railway.app](https://paewall.railway.app)  
Results in < 10 seconds · FastAPI + Jinja2 · Deployed on Railway

---

## Slide 5 — Results & Next Steps

# What We Learned + What's Next

**Main findings:**
- Fine-tuning on litigation pairs gives the largest R@10 gains (+36% over classical)
- Software vertical benefits most (R@10 = 53.4%); industrial fails — sparse training data
- Primary failure mode: vocabulary mismatch (legal claim language ≠ marketing 10-K text)
- Classical ML dominates on ranking precision → cross-encoder re-ranker is the next lever

**Ablation hypothesis confirmed (3 of 4 conditions):**  
BM25 (13%) → Classical (24.1%) → Fine-tuned DE (32.8%)  
Zero-shot PatentSBERTa pending — expected to land between classical and fine-tuned

**Future work:**
- Multimodal: add SigLIP patent figure embeddings to product tower
- Cross-encoder re-ranker on top-20 dual-encoder candidates (target: close MRR gap)
- Expand PAE-Bench to 2,000+ pairs (ITC proceedings + PTAB records)
- Jurisdiction-aware enforcement probability with case outcome data

**Thank you!**  
Code: github.com/Shreya-Mendi/PAEwall
