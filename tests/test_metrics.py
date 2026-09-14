"""
Unit tests for CER, WER, and Levenshtein edit distance calculations.
"""

from indic_pipeline.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
    evaluate_prediction,
)


def test_perfect_match():
    ref = "ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ"
    hyp = "ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ"

    cer, char_edits = calculate_cer(ref, hyp)
    wer, word_edits = calculate_wer(ref, hyp)

    assert cer == 0.0
    assert wer == 0.0
    assert char_edits.substitutions == 0
    assert char_edits.deletions == 0
    assert char_edits.insertions == 0
    assert word_edits.substitutions == 0


def test_substitution_and_deletion():
    ref = "ಕರ್ನಾಟಕ ರಾಜ್ಯ"  # 12 chars + space
    hyp = "ಕರ್ನಾಟಗ ರಾಜ"    # 'ಕ' replaced by 'ಗ' (substitution), '್ಯ' deleted (deletion)

    cer, char_edits = calculate_cer(ref, hyp)
    assert char_edits.substitutions == 1
    assert char_edits.deletions == 2
    assert cer > 0.0


def test_evaluation_metrics_summary():
    ref = "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೆ"
    hyp = "ನಮಸ್ಕಾರ ಮಿತ್ರರೆ"

    metrics = evaluate_prediction(ref, hyp)
    assert metrics.cer > 0.0
    assert metrics.wer == 0.5  # 1 word match, 1 word substituted out of 2
    assert metrics.word_accuracy == 50.0
    assert metrics.char_accuracy < 100.0


def test_empty_hypothesis():
    ref = "ಪರೀಕ್ಷೆ"
    hyp = ""

    metrics = evaluate_prediction(ref, hyp)
    assert metrics.cer == 1.0
    assert metrics.wer == 1.0
    assert metrics.char_edits.deletions == len(ref)
