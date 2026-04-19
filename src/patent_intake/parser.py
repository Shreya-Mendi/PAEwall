"""
Patent claim parsing and structuring.

Extracts independent claims from patent text, decomposes them into
structured elements (preamble, transitional phrase, limitations),
and classifies CPC subclass.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ClaimElement:
    """A single element of a patent claim."""
    element_type: str  # "preamble", "transition", "limitation"
    text: str
    index: int


@dataclass
class ParsedClaim:
    """A fully parsed independent claim."""
    claim_number: int
    full_text: str
    elements: list[ClaimElement] = field(default_factory=list)
    is_independent: bool = True


class PatentParser:
    """Parses patent documents into structured claim representations."""

    _TRANSITION_PHRASES = [
        "consisting essentially of",
        "consisting of",
        "comprising",
        "including",
        "having",
        "characterized by",
        "wherein",
    ]
    _DEPENDENT_RE = re.compile(r"\bclaim\s+\d+\b", re.IGNORECASE)

    def extract_claims(self, patent_text: str) -> list[str]:
        """Extract individual claims from full patent text."""
        # Isolate the CLAIMS section header (must look like a section break, not an inline reference)
        section_match = re.split(
            r'(?:^|\n)\s*(?:WHAT IS CLAIMED IS|WHAT IS CLAIMED|THE CLAIMS ARE|CLAIMS)\s*[:\n]',
            patent_text, maxsplit=1, flags=re.IGNORECASE
        )
        text = section_match[1] if len(section_match) > 1 else patent_text

        # Split on numbered claim lines like "1." or "\n1."
        parts = re.split(r'(?:^|\n)\s*(\d+)\.\s+', text)
        claims = []
        for i in range(1, len(parts) - 1, 2):
            num = parts[i].strip()
            body = parts[i + 1].strip()
            if body:
                # Remove continuation lines with just whitespace / page headers
                body = re.sub(r'\n\s{10,}', ' ', body)
                claims.append(f"{num}. {body}")
        return claims

    def classify_independence(self, claim_text: str) -> bool:
        """Determine if a claim is independent (True) or dependent (False)."""
        return not bool(self._DEPENDENT_RE.search(claim_text))

    def decompose_claim(self, claim_text: str) -> ParsedClaim:
        """
        Decompose a claim into preamble, transitional phrase, and limitations.
        Uses rule-based parsing; falls back to treating whole text as preamble.
        """
        m = re.match(r'^(\d+)\.\s+', claim_text)
        claim_num = int(m.group(1)) if m else 0
        body = claim_text[m.end():] if m else claim_text
        is_independent = self.classify_independence(claim_text)
        elements: list[ClaimElement] = []

        # Find earliest transition phrase
        transition_pos = None
        matched_phrase = None
        for phrase in self._TRANSITION_PHRASES:
            pos = body.lower().find(phrase)
            if pos != -1 and (transition_pos is None or pos < transition_pos):
                transition_pos = pos
                matched_phrase = phrase

        if transition_pos is not None:
            preamble = body[:transition_pos].strip()
            remainder = body[transition_pos + len(matched_phrase):].strip()

            if preamble:
                elements.append(ClaimElement("preamble", preamble, 0))
            elements.append(ClaimElement("transition", matched_phrase, len(elements)))

            # Limitations are separated by semicolons or explicit line breaks
            raw = re.split(r';\s*|\n+', remainder)
            for lim in raw:
                lim = lim.strip().rstrip(';').strip()
                if lim:
                    elements.append(ClaimElement("limitation", lim, len(elements)))
        else:
            elements.append(ClaimElement("preamble", body, 0))

        return ParsedClaim(
            claim_number=claim_num,
            full_text=claim_text,
            elements=elements,
            is_independent=is_independent,
        )

    def parse_independent_claims(self, patent_text: str) -> list[ParsedClaim]:
        """Extract and parse all independent claims from a patent."""
        claims = self.extract_claims(patent_text)
        parsed = []
        for c in claims:
            pc = self.decompose_claim(c)
            if pc.is_independent:
                parsed.append(pc)
        return parsed


class CPCClassifier:
    """
    Classifies patents by technology vertical using CPC codes or keyword fallback.
    When CPC codes are available (from patent data), uses them directly.
    Falls back to keyword scoring when only claim text is available.
    """

    _CPC_PREFIX_MAP = {
        "G16H": "medical_devices",
        "A61": "medical_devices",
        "G06Q": "fintech",
        "G06": "software",
        "H04": "consumer_electronics",
        "H01": "consumer_electronics",
        "B": "industrial",
        "F": "industrial",
    }

    _KEYWORD_MAP = {
        "software": ["software", "computer", "processor", "algorithm", "data", "server", "network", "interface", "application"],
        "consumer_electronics": ["wireless", "mobile", "bluetooth", "display", "signal", "audio", "video", "antenna", "receiver"],
        "medical_devices": ["patient", "medical", "diagnosis", "treatment", "clinical", "drug", "therapeutic", "sensor", "implant"],
        "fintech": ["payment", "transaction", "financial", "bank", "currency", "ledger", "authentication", "wallet"],
        "industrial": ["mechanical", "motor", "fluid", "pressure", "temperature", "valve", "pump", "turbine", "structural"],
    }

    def predict_from_cpc(self, cpc_codes: list[str]) -> str:
        """Return vertical from CPC codes. Returns 'other' if no match."""
        for code in cpc_codes:
            for prefix, vertical in self._CPC_PREFIX_MAP.items():
                if code.startswith(prefix):
                    return vertical
        return "other"

    def predict(self, claim_text: str) -> dict[str, float]:
        """Keyword-based vertical scoring when CPC codes are unavailable."""
        text = claim_text.lower()
        words = text.split()
        n = max(len(words), 1)
        scores = {
            v: sum(text.count(w) for w in kws) / n
            for v, kws in self._KEYWORD_MAP.items()
        }
        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 4) for k, v in scores.items()}
