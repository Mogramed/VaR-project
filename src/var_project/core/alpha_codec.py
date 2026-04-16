from __future__ import annotations

import re

_ALPHA_TOKEN_RE = re.compile(r"^\d+(?:p\d+)?$")


def encode_alpha_token(alpha: float) -> str:
    """Encode alpha into a stable suffix token used in column names.

    Examples:
    - 0.95 -> "95"
    - 0.975 -> "97p5"
    - 0.99 -> "99"
    """
    alpha_value = float(alpha)
    percent = alpha_value * 100.0
    rounded_percent = round(percent)
    if abs(percent - rounded_percent) <= 1e-9:
        return str(int(rounded_percent))
    text = f"{percent:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def decode_alpha_token(token: str) -> float:
    normalized = str(token).strip().lower()
    if not _ALPHA_TOKEN_RE.fullmatch(normalized):
        raise ValueError(f"Invalid alpha token: {token!r}")
    percent_text = normalized.replace("p", ".")
    return float(percent_text) / 100.0


def alpha_lookup_tokens(alpha: float) -> list[str]:
    """Return preferred and legacy alpha tokens for lookup compatibility."""
    preferred = encode_alpha_token(alpha)
    legacy = str(int(round(float(alpha) * 100)))
    if legacy == preferred:
        return [preferred]
    return [preferred, legacy]
