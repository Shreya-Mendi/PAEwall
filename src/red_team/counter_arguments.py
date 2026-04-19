"""
Red-team counter-argument generation.

For every claim chart, generates:
1. Top non-infringement arguments a defendant would raise
2. Top invalidity risks via prior art
3. Enforcement win probability estimate

Uses Claude when ANTHROPIC_API_KEY is set; otherwise generates template-based
arguments from the claim chart data so the demo runs without any API key.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from llm_client import chat as _llm_chat


@dataclass
class CounterArgument:
    """A single counter-argument against infringement."""
    argument_type: str  # "non_infringement" | "invalidity"
    summary: str
    legal_basis: str
    cited_precedent: str | None = None
    strength: float = 0.5


@dataclass
class RedTeamReport:
    """Complete red-team analysis for a (patent, product) pair."""
    patent_id: str
    product_id: str
    non_infringement_args: list[CounterArgument] = field(default_factory=list)
    invalidity_args: list[CounterArgument] = field(default_factory=list)
    enforcement_probability: float = 0.5


_SYSTEM_NON_INFRINGEMENT = """\
You are a defense-side patent attorney. Given a claim chart mapping patent limitations to product features, \
generate the top non-infringement arguments the defendant would raise.

Output ONLY a JSON array of objects with:
- "summary": one-sentence argument
- "legal_basis": e.g. "missing limitation", "claim differentiation", "prosecution history estoppel"
- "cited_precedent": relevant case name or null
- "strength": float 0.0-1.0 (how strong this argument is likely to be)
"""

_SYSTEM_INVALIDITY = """\
You are a defense-side patent attorney. Given patent claims and their technology area, \
generate invalidity arguments under 35 U.S.C. §102 (anticipation) and §103 (obviousness).

Output ONLY a JSON array of objects with:
- "summary": one-sentence argument describing the prior art challenge
- "legal_basis": "§102 anticipation" | "§103 obviousness" | "§101 patent-eligible subject matter" | "§112 indefiniteness"
- "cited_precedent": relevant case name or null
- "strength": float 0.0-1.0
"""


def _llm_available(client=None) -> bool:
    return bool(os.environ.get("DUKE_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _rule_non_infringement(claim_chart: dict, num_args: int) -> list[CounterArgument]:
    """Template-based non-infringement arguments derived from claim chart labels."""
    mappings = claim_chart.get("mappings", [])
    product = claim_chart.get("product_id", "the defendant")

    unmet = [m for m in mappings if m.get("faithfulness_label") == "does_not_support"]
    partial = [m for m in mappings if m.get("faithfulness_label") == "partially_supports"]

    args = []
    if unmet:
        lim_excerpt = unmet[0].get("limitation", "a claim element")[:80]
        args.append(CounterArgument(
            argument_type="non_infringement",
            summary=f"{product}'s products do not practice the limitation '{lim_excerpt}…', which is absent from their documented features.",
            legal_basis="missing limitation",
            cited_precedent="Freedman Seating Co. v. Am. Seating Co., 420 F.3d 1350 (Fed. Cir. 2005)",
            strength=0.75,
        ))
    if partial:
        lim_excerpt = partial[0].get("limitation", "a claim element")[:80]
        args.append(CounterArgument(
            argument_type="non_infringement",
            summary=f"Even if {product}'s implementation touches on '{lim_excerpt}…', it differs in a substantial way that avoids the claim scope.",
            legal_basis="claim differentiation",
            cited_precedent="Curtiss-Wright Flow Control Corp. v. Velan, Inc., 438 F.3d 1374 (Fed. Cir. 2006)",
            strength=0.60,
        ))
    args.append(CounterArgument(
        argument_type="non_infringement",
        summary=f"The patent's prosecution history shows the applicant narrowed the claims to overcome prior art, excluding {product}'s implementation under prosecution history estoppel.",
        legal_basis="prosecution history estoppel",
        cited_precedent="Festo Corp. v. Shoketsu Kinzoku Kogyo Kabushiki Co., 535 U.S. 722 (2002)",
        strength=0.55,
    ))
    args.append(CounterArgument(
        argument_type="non_infringement",
        summary=f"{product} employs a materially different design-around that does not infringe even under the doctrine of equivalents.",
        legal_basis="doctrine of equivalents — insubstantial differences",
        cited_precedent="Warner-Jenkinson Co. v. Hilton Davis Chemical Co., 520 U.S. 17 (1997)",
        strength=0.50,
    ))
    return args[:num_args]


def _rule_invalidity(patent_claims: list[str], priority_date: str, num_args: int) -> list[CounterArgument]:
    """Template-based invalidity arguments."""
    claim_text = " ".join(patent_claims[:2]).lower()
    is_method = "method" in claim_text or "process" in claim_text
    is_software = any(w in claim_text for w in ["computer", "processor", "software", "network", "server", "data"])

    args = []
    if is_software:
        args.append(CounterArgument(
            argument_type="invalidity",
            summary="The asserted claims recite abstract ideas (data processing / mathematical concepts) implemented on generic computer hardware, failing the Alice two-step framework.",
            legal_basis="§101 patent-eligible subject matter",
            cited_precedent="Alice Corp. v. CLS Bank Int'l, 573 U.S. 208 (2014)",
            strength=0.70,
        ))
    args.append(CounterArgument(
        argument_type="invalidity",
        summary=f"Prior art published before the priority date of {priority_date or 'the patent'} anticipates each element of the asserted claims under §102.",
        legal_basis="§102 anticipation",
        cited_precedent=None,
        strength=0.65,
    ))
    if is_method:
        args.append(CounterArgument(
            argument_type="invalidity",
            summary="The claimed method would have been obvious to a person of ordinary skill in the art by combining well-known techniques documented in prior art before the priority date.",
            legal_basis="§103 obviousness",
            cited_precedent="KSR Int'l Co. v. Teleflex Inc., 550 U.S. 398 (2007)",
            strength=0.60,
        ))
    else:
        args.append(CounterArgument(
            argument_type="invalidity",
            summary="The claim terms lack definite boundaries rendering them indefinite and unenforceable.",
            legal_basis="§112 indefiniteness",
            cited_precedent="Nautilus, Inc. v. Biosig Instruments, Inc., 572 U.S. 898 (2014)",
            strength=0.50,
        ))
    return args[:num_args]


def _parse_llm_args(raw: str, arg_type: str) -> list[CounterArgument]:
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip()).rstrip("` \n")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        items = json.loads(m.group()) if m else []
    return [
        CounterArgument(
            argument_type=arg_type,
            summary=item.get("summary", ""),
            legal_basis=item.get("legal_basis", ""),
            cited_precedent=item.get("cited_precedent"),
            strength=float(item.get("strength", 0.5)),
        )
        for item in items
    ]


class NonInfringementGenerator:
    """Generates non-infringement arguments; LLM when available, rule-based otherwise."""

    def __init__(self, llm_client=None):
        self.client = llm_client

    def generate(self, claim_chart: dict, num_args: int = 3) -> list[CounterArgument]:
        if _llm_available(self.client):
            try:
                return self._llm_generate(claim_chart, num_args)
            except Exception:
                pass
        return _rule_non_infringement(claim_chart, num_args)

    def _llm_generate(self, claim_chart: dict, num_args: int) -> list[CounterArgument]:
        mappings_summary = [
            {
                "limitation": m.get("limitation", "")[:200],
                "evidence_found": m.get("faithfulness_label", "does_not_support"),
                "evidence": m.get("evidence", "")[:200],
            }
            for m in claim_chart.get("mappings", [])
        ]
        user_msg = (
            f"Claim chart for patent {claim_chart.get('patent_id', 'unknown')} "
            f"vs product {claim_chart.get('product_id', 'unknown')}:\n\n"
            f"{json.dumps(mappings_summary, indent=2)}\n\n"
            f"Generate {num_args} non-infringement arguments. Return a JSON array only."
        )
        raw = _llm_chat(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM_NON_INFRINGEMENT,
            max_tokens=1024,
        )
        if raw is None:
            raise RuntimeError("LLM unavailable")
        return _parse_llm_args(raw, "non_infringement")


class InvalidityGenerator:
    """Generates invalidity arguments; LLM when available, rule-based otherwise."""

    def __init__(self, llm_client=None):
        self.client = llm_client

    def generate(self, patent_claims: list[str], priority_date: str, num_args: int = 3) -> list[CounterArgument]:
        if _llm_available(self.client):
            try:
                return self._llm_generate(patent_claims, priority_date, num_args)
            except Exception:
                pass
        return _rule_invalidity(patent_claims, priority_date, num_args)

    def _llm_generate(self, patent_claims: list[str], priority_date: str, num_args: int) -> list[CounterArgument]:
        claims_text = "\n\n".join(c[:400] for c in patent_claims[:3])
        user_msg = (
            f"Patent priority date: {priority_date}\n\n"
            f"Independent claims:\n{claims_text}\n\n"
            f"Generate {num_args} invalidity arguments. Return a JSON array only."
        )
        raw = _llm_chat(
            messages=[{"role": "user", "content": user_msg}],
            system=_SYSTEM_INVALIDITY,
            max_tokens=1024,
        )
        if raw is None:
            raise RuntimeError("LLM unavailable")
        return _parse_llm_args(raw, "invalidity")


class EnforcementProbabilityModel:
    """
    Estimates enforcement win probability.

    Uses a heuristic formula based on faithfulness scores until
    an XGBoost model is trained on outcome data.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path:
            self._load(model_path)

    def _load(self, path: str):
        try:
            import pickle
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        except Exception:
            self.model = None

    def predict(self, features: dict) -> float:
        if self.model is not None:
            try:
                import numpy as np
                feature_vector = self._build_vector(features)
                return float(self.model.predict_proba([feature_vector])[0, 1])
            except Exception:
                pass
        return self._heuristic(features)

    def _heuristic(self, features: dict) -> float:
        faithfulness = features.get("faithfulness_score", 0.5)
        met = features.get("num_limitations_met", 0)
        total = max(features.get("num_limitations_total", 1), 1)
        coverage = met / total
        ni_strength = features.get("non_infringement_strength", 0.5)
        inv_strength = features.get("invalidity_strength", 0.5)
        defense_risk = (ni_strength + inv_strength) / 2
        raw = (0.5 * faithfulness + 0.3 * coverage + 0.2 * (1 - defense_risk))
        return round(min(max(raw, 0.0), 1.0), 3)

    def _build_vector(self, features: dict) -> list:
        return [
            features.get("faithfulness_score", 0.5),
            features.get("num_limitations_met", 0),
            features.get("num_limitations_total", 1),
            features.get("citation_count", 0),
            features.get("non_infringement_strength", 0.5),
            features.get("invalidity_strength", 0.5),
        ]
