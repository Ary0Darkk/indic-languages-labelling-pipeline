"""
CER (Character Error Rate) and WER (Word Error Rate) metrics calculation.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein
from indic_pipeline.evaluation.normalizer import IndicNormalizer, default_normalizer


class EditBreakdown(BaseModel):
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    matches: int = 0
    total_reference: int = 0


class EvaluationMetrics(BaseModel):
    reference_length: int
    prediction_length: int
    cer: float
    wer: float
    char_accuracy: float
    word_accuracy: float
    char_edits: EditBreakdown
    word_edits: EditBreakdown
    normalized_reference: str
    normalized_prediction: str


def compute_edit_ops(reference_seq: list, hypothesis_seq: list) -> EditBreakdown:
    """Computes Levenshtein edit operations breakdown between two sequences."""
    ref_len = len(reference_seq)
    hyp_len = len(hypothesis_seq)

    if ref_len == 0 and hyp_len == 0:
        return EditBreakdown(total_reference=0)
    if ref_len == 0:
        return EditBreakdown(insertions=hyp_len, total_reference=0)
    if hyp_len == 0:
        return EditBreakdown(deletions=ref_len, total_reference=ref_len)

    # Use rapidfuzz editops
    editops = Levenshtein.editops(reference_seq, hypothesis_seq)
    subs = sum(1 for op in editops if op[0] == "replace")
    dels = sum(1 for op in editops if op[0] == "delete")
    ins = sum(1 for op in editops if op[0] == "insert")
    matches = ref_len - (subs + dels)

    return EditBreakdown(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        matches=max(0, matches),
        total_reference=ref_len,
    )


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalizer: Optional[IndicNormalizer] = None,
) -> Tuple[float, EditBreakdown]:
    """
    Calculates Character Error Rate (CER) = (S + D + I) / N
    Returns (cer_value, edit_breakdown).
    """
    norm = normalizer or default_normalizer
    norm_ref = norm.normalize(reference)
    norm_hyp = norm.normalize(hypothesis)

    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)

    edits = compute_edit_ops(ref_chars, hyp_chars)
    if edits.total_reference == 0:
        cer = 0.0 if len(hyp_chars) == 0 else 1.0
    else:
        cer = (edits.substitutions + edits.deletions + edits.insertions) / edits.total_reference

    return cer, edits


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalizer: Optional[IndicNormalizer] = None,
) -> Tuple[float, EditBreakdown]:
    """
    Calculates Word Error Rate (WER) = (S_w + D_w + I_w) / N_w
    Returns (wer_value, edit_breakdown).
    """
    norm = normalizer or default_normalizer
    norm_ref = norm.normalize(reference)
    norm_hyp = norm.normalize(hypothesis)

    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()

    edits = compute_edit_ops(ref_words, hyp_words)
    if edits.total_reference == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = (edits.substitutions + edits.deletions + edits.insertions) / edits.total_reference

    return wer, edits


def evaluate_prediction(
    reference: str,
    hypothesis: str,
    normalizer: Optional[IndicNormalizer] = None,
) -> EvaluationMetrics:
    """Full comprehensive evaluation returning CER, WER, accuracy, and normalized texts."""
    norm = normalizer or default_normalizer
    norm_ref = norm.normalize(reference)
    norm_hyp = norm.normalize(hypothesis)

    cer, char_edits = calculate_cer(norm_ref, norm_hyp, normalizer=norm)
    wer, word_edits = calculate_wer(norm_ref, norm_hyp, normalizer=norm)

    char_acc = max(0.0, 1.0 - cer) * 100.0
    word_acc = max(0.0, 1.0 - wer) * 100.0

    return EvaluationMetrics(
        reference_length=len(norm_ref),
        prediction_length=len(norm_hyp),
        cer=round(cer, 4),
        wer=round(wer, 4),
        char_accuracy=round(char_acc, 2),
        word_accuracy=round(word_acc, 2),
        char_edits=char_edits,
        word_edits=word_edits,
        normalized_reference=norm_ref,
        normalized_prediction=norm_hyp,
    )
