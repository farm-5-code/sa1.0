"""Минимальный генератор отчётов.

Этот модуль оставлен как заглушка: в текущей версии проекта отчёты
рендерятся в консоль (main.py) и в Streamlit (dashboard/app.py).
"""

from __future__ import annotations
from typing import Dict


def build_text_report(result: Dict) -> str:
    """Собирает простой текстовый отчёт из результата анализа."""
    lines = []
    lines.append(result.get("match", "Match"))
    conf = result.get("confidence", 0)
    lines.append(f"Confidence: {conf}%")
    for rec in result.get("recommendations", []):
        lines.append(f"- {rec}")
    return "\n".join(lines)
