"""
Unit tests for Indic Unicode Normalization.
"""

import unicodedata
from indic_pipeline.evaluation.normalizer import IndicNormalizer


def test_unicode_nfc_normalization():
    normalizer = IndicNormalizer()
    # Kannada word with decomposed vs composed characters
    text_nfd = unicodedata.normalize("NFD", "ಕರ್ನಾಟಕ")
    text_nfc = unicodedata.normalize("NFC", "ಕರ್ನಾಟಕ")

    norm_nfd = normalizer.normalize(text_nfd)
    norm_nfc = normalizer.normalize(text_nfc)

    assert norm_nfd == norm_nfc
    assert norm_nfd == "ಕರ್ನಾಟಕ"


def test_zwnj_and_zwj_handling():
    normalizer = IndicNormalizer()
    # Kannada word with ZWNJ \u200C and ZWJ \u200D
    text_with_zwnj = "ರಾಜ್ಯ\u200Cಸರ್ಕಾರ"
    text_with_zwj = "ರಾಜ್ಯ\u200Dಸರ್ಕಾರ"
    plain_text = "ರಾಜ್ಯಸರ್ಕಾರ"

    assert normalizer.normalize(text_with_zwnj) == plain_text
    assert normalizer.normalize(text_with_zwj) == plain_text


def test_digit_normalization():
    normalizer = IndicNormalizer()
    # Kannada digits: ೧೨೩೪೫ (12345)
    kannada_digits = "ದಿನಾಂಕ: ೧೫-೦೮-೨೦೨೪"
    normalized = normalizer.normalize(kannada_digits, convert_digits=True)
    assert normalized == "ದಿನಾಂಕ: 15-08-2024"

    # Hindi/Devanagari digits: १२३४५
    hindi_digits = "दिनांक: १५-०८-२०२४"
    assert normalizer.normalize(hindi_digits, convert_digits=True) == "दिनांक: 15-08-2024"


def test_whitespace_and_quote_normalization():
    normalizer = IndicNormalizer()
    raw = "   ‘ಕರ್ನಾಟಕ’   ರಾಜ್ಯ    \n\n\n  ಸರ್ಕಾರ   "
    expected = "'ಕರ್ನಾಟಕ' ರಾಜ್ಯ\nಸರ್ಕಾರ"
    assert normalizer.normalize(raw) == expected
