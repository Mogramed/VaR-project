from __future__ import annotations

from var_project.core.alpha_codec import alpha_lookup_tokens, decode_alpha_token, encode_alpha_token


def test_alpha_codec_encodes_fractional_percent_with_p_separator():
    assert encode_alpha_token(0.95) == "95"
    assert encode_alpha_token(0.975) == "97p5"
    assert encode_alpha_token(0.99) == "99"


def test_alpha_codec_decodes_tokens():
    assert decode_alpha_token("95") == 0.95
    assert decode_alpha_token("97p5") == 0.975


def test_alpha_codec_lookup_tokens_include_legacy_rounding():
    assert alpha_lookup_tokens(0.95) == ["95"]
    assert alpha_lookup_tokens(0.975) == ["97p5", "98"]
