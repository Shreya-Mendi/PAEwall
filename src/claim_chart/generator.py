"""
Claim chart generation with faithfulness verification.

Generates claim charts mapping patent claim limitations to product features,
then scores each mapping with a faithfulness verifier.

LLM (Claude) is used when ANTHROPIC_API_KEY is set; otherwise falls back to
keyword-overlap rule-based generation so the demo runs without any API key.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from llm_client import chat as _llm_chat


@dataclass
class ClaimMapping:
    """A single limitation-to-evidence mapping in a claim chart."""
    limitation: str
    evidence: str
    source: str = "product_description"
    faithfulness_score: float = 0.0
    faithfulness_label: str = "does_not_support"  # supports | partially_supports | does_not_support


@dataclass
class ClaimChart:
    """A complete claim chart for a (patent, product) pair."""
    patent_id: str
    product_id: str
    mappings: list[ClaimMapping] = field(default_factory=list)
    overall_confidence: float = 0.0


_SYSTEM_CLAIM_CHART = """\
You are a patent infringement analyst. Given a list of patent claim limitations and product description evidence, \
generate a structured claim chart. For each limitation, identify the best matching evidence snippet from the product \
description and assess whether it supports the limitation.

Output ONLY a JSON array — no markdown fences, no explanation. Each element must have these keys:
- "limitation": the exact claim limitation text
- "evidence": the most relevant excerpt from the product description (or "" if none)
- "faithfulness_label": one of "supports", "partially_supports", "does_not_support"
- "faithfulness_score": float 0.0-1.0 (1.0 = clearly supports, 0.0 = clearly does not)
"""

_SYSTEM_FAITHFULNESS = """\
You are a patent claim faithfulness verifier. Given a claim limitation and an evidence snippet from a product description, \
determine whether the evidence supports the limitation.

Output ONLY a JSON object with:
- "label": "supports" | "partially_supports" | "does_not_support"
- "score": float 0.0-1.0
- "reason": one sentence explanation
"""


def _tokenize(text: str) -> set[str]:
    stopwords = {"a", "an", "the", "of", "to", "and", "or", "in", "is", "are",
                 "for", "with", "that", "this", "by", "at", "from", "on", "be",
                 "as", "it", "its", "at", "which", "were", "has", "have", "been"}
    return {w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if w not in stopwords and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _best_sentence(limitation: str, product_text: str) -> tuple[str, float]:
    """Return (best_sentence, jaccard_score) from product_text for a given limitation."""
    lim_toks = _tokenize(limitation)
    sentences = re.split(r"[.!?]\s+", product_text)
    best_sent, best_score = "", 0.0
    for sent in sentences:
        s = sent.strip()
        if len(s) < 20:
            continue
        score = _jaccard(lim_toks, _tokenize(s))
        if score > best_score:
            best_score, best_sent = score, s
    return best_sent, best_score


def _rule_based_generate(parsed_claims: list, product_evidence: list[str]) -> list[ClaimMapping]:
    """Keyword-overlap claim chart generation — no LLM required."""
    product_text = " ".join(product_evidence)

    limitations = []
    for claim in parsed_claims:
        for elem in claim.elements:
            if elem.element_type == "limitation":
                limitations.append(elem.text)
    if not limitations:
        for claim in parsed_claims:
            limitations.append(claim.full_text[:500])

    mappings = []
    for lim in limitations[:8]:
        evidence, score = _best_sentence(lim, product_text)
        if score >= 0.12:
            label, fscore = "supports", min(0.5 + score * 3.5, 0.95)
        elif score >= 0.05:
            label, fscore = "partially_supports", 0.25 + score * 3.0
        else:
            label, fscore = "does_not_support", max(score * 2, 0.05)
        mappings.append(ClaimMapping(
            limitation=lim,
            evidence=evidence if score >= 0.05 else "",
            source="product_description",
            faithfulness_score=round(fscore, 3),
            faithfulness_label=label,
        ))
    return mappings


class ClaimChartGenerator:
    """Generates claim charts; uses Claude when available, rule-based otherwise."""

    def __init__(self, llm_client=None):
        self.client = llm_client
        self._llm_available = self._check_llm(llm_client)

    @staticmethod
    def _check_llm(client) -> bool:
        return bool(os.environ.get("DUKE_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, parsed_claims: list, product_evidence: list[str]) -> list[ClaimMapping]:
        if self._llm_available and self.client is not None:
            try:
                return self._llm_generate(parsed_claims, product_evidence)
            except Exception:
                pass
        return _rule_based_generate(parsed_claims, product_evidence)

    def _llm_generate(self, parsed_claims: list, product_evidence: list[str]) -> list[ClaimMapping]:
        limitations = []
        for claim in parsed_claims:
            for elem in claim.elements:
                if elem.element_type == "limitation":
                    limitations.append(elem.text)
        if not limitations:
            for claim in parsed_claims:
                limitations.append(claim.full_text[:500])

        evidence_text = "\n\n---\n\n".join(e[:800] for e in product_evidence[:8])
        user_msg = (
            f"Claim limitations:\n{json.dumps(limitations, indent=2)}\n\n"
            f"Product description evidence:\n{evidence_text}\n\n"
            "Return a JSON array only."
        )

        raw = _llm_chat(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM_CLAIM_CHART,
            max_tokens=2048,
        )
        if raw is None:
            raise RuntimeError("LLM unavailable")

        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip()).rstrip("` \n")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            items = json.loads(m.group()) if m else []

        return [
            ClaimMapping(
                limitation=item.get("limitation", ""),
                evidence=item.get("evidence", ""),
                source="product_description",
                faithfulness_score=float(item.get("faithfulness_score", 0.0)),
                faithfulness_label=item.get("faithfulness_label", "does_not_support"),
            )
            for item in items
        ]


class FaithfulnessVerifier:
    """NLI-style faithfulness verifier; falls back to keyword overlap when no LLM."""

    def __init__(self, llm_client=None):
        self.client = llm_client
        self._llm_available = ClaimChartGenerator._check_llm(llm_client)

    def verify(self, limitation: str, evidence: str) -> tuple[str, float]:
        if not evidence.strip():
            return "does_not_support", 0.0

        if self._llm_available and self.client is not None:
            try:
                return self._llm_verify(limitation, evidence)
            except Exception:
                pass

        score = _jaccard(_tokenize(limitation), _tokenize(evidence))
        if score >= 0.12:
            return "supports", min(0.5 + score * 3.5, 0.95)
        elif score >= 0.05:
            return "partially_supports", 0.25 + score * 3.0
        return "does_not_support", max(score * 2, 0.05)

    def _llm_verify(self, limitation: str, evidence: str) -> tuple[str, float]:
        user_msg = f"Limitation: {limitation}\n\nEvidence: {evidence[:1000]}\n\nReturn JSON only."
        raw = _llm_chat(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM_FAITHFULNESS,
            max_tokens=256,
            json_mode=True,
        )
        if raw is None:
            raise RuntimeError("LLM unavailable")
        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip()).rstrip("` \n")
        result = json.loads(raw)
        return result.get("label", "does_not_support"), float(result.get("score", 0.0))

    def score_chart(self, chart: ClaimChart) -> ClaimChart:
        scored = []
        for mapping in chart.mappings:
            label, score = self.verify(mapping.limitation, mapping.evidence)
            scored.append(ClaimMapping(
                limitation=mapping.limitation,
                evidence=mapping.evidence,
                source=mapping.source,
                faithfulness_score=score,
                faithfulness_label=label,
            ))
        chart.mappings = scored
        if scored:
            chart.overall_confidence = sum(m.faithfulness_score for m in scored) / len(scored)
        return chart
