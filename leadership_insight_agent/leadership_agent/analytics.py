from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from .retriever import RetrievedChunk


@dataclass
class SeriesPoint:
    label: str
    value: float
    citation: str


def _find_money_values(text: str) -> List[Tuple[str, float]]:
    """Extract simple money-like numbers. Intentionally basic.

    Recognizes patterns like: 12.3B, 4.5 million, $123,456, etc.
    """
    patterns = [
        r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s?(?:billion|B)\b",
        r"([0-9]+(?:\.[0-9]+)?)\s?(?:million|M)\b",
    ]
    out: List[Tuple[str, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw = m.group(1)
            try:
                val = float(raw.replace(",", ""))
            except Exception:
                continue
            token = m.group(0)
            # Scale if million/billion
            if re.search(r"billion|\bB\b", token, re.IGNORECASE):
                val = val * 1e9
            elif re.search(r"million|\bM\b", token, re.IGNORECASE):
                val = val * 1e6
            out.append((token, val))
    return out


def extract_revenue_series(chunks: List[RetrievedChunk]) -> List[SeriesPoint]:
    """Try to build a time series by looking for quarter/year labels near numbers.

    This is heuristic and may fail depending on document formatting.
    """
    points: List[SeriesPoint] = []
    # Look for Q1/Q2/Q3/Q4 or FY or year
    label_pat = re.compile(r"\b(Q[1-4]\s?\d{2,4}|FY\s?\d{2,4}|\d{4})\b", re.IGNORECASE)

    for ch in chunks:
        labels = label_pat.findall(ch.text)
        vals = _find_money_values(ch.text)
        if not labels or not vals:
            continue
        # Pair first label with first value for simplicity
        label = labels[0].replace(" ", "")
        token, val = vals[0]
        points.append(SeriesPoint(label=label, value=val, citation=ch.chunk_id))

    # Deduplicate by label keeping max value
    best = {}
    for p in points:
        if p.label not in best or p.value > best[p.label].value:
            best[p.label] = p

    # Sort labels: try year then quarter
    def sort_key(lbl: str):
        # FY2024, 2024, Q32024, Q3'24
        m_year = re.search(r"(\d{4})", lbl)
        year = int(m_year.group(1)) if m_year else 0
        m_q = re.search(r"Q([1-4])", lbl, re.IGNORECASE)
        q = int(m_q.group(1)) if m_q else 0
        return (year, q)

    return sorted(best.values(), key=lambda p: sort_key(p.label))


def plot_series(points: List[SeriesPoint], title: str, out_path: str) -> Optional[str]:
    if len(points) < 2:
        return None

    df = pd.DataFrame({"label": [p.label for p in points], "value": [p.value for p in points]})
    plt.figure()
    plt.plot(df["label"], df["value"], marker="o")
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
