"""
test_spam_classifier.py

This is the actual point of the project: a Machine Learning model is not
"done" just because it trains. It needs to be tested like any other
software component -- plus a few checks that are specific to ML systems
(determinism, robustness to input variation, and threshold-based accuracy
instead of exact-match assertions).

Run with:  pytest test_spam_classifier.py -v
"""

import pytest
from spam_classifier import train_model, predict

# Train once and reuse across tests (fixture) so we're not retraining
# the model for every single test case.
@pytest.fixture(scope="module")
def trained_model():
    vectorizer, model, accuracy = train_model()
    return vectorizer, model, accuracy


# ---------------------------------------------------------------------
# 1. Regression test: accuracy must not silently degrade below a floor.
#    In real ML pipelines this is what stops a bad retrain from shipping.
# ---------------------------------------------------------------------
def test_accuracy_meets_minimum_threshold(trained_model):
    _, _, accuracy = trained_model
    MINIMUM_ACCEPTABLE_ACCURACY = 0.70
    assert accuracy >= MINIMUM_ACCEPTABLE_ACCURACY, (
        f"Model accuracy {accuracy:.2%} fell below the "
        f"{MINIMUM_ACCEPTABLE_ACCURACY:.0%} floor"
    )


# ---------------------------------------------------------------------
# 2. Output contract test: prediction must always be a valid label (0 or 1).
#    Guards against the model or preprocessing silently returning garbage.
# ---------------------------------------------------------------------
def test_prediction_output_is_valid_label(trained_model):
    vectorizer, model, _ = trained_model
    result = predict(vectorizer, model, "Free money click now")
    assert result in (0, 1), f"Expected 0 or 1, got {result}"


# ---------------------------------------------------------------------
# 3. Determinism test: same input must give the same output every time.
#    ML pipelines can accidentally become non-deterministic (e.g. if a
#    random seed is missing somewhere) -- this catches that class of bug.
# ---------------------------------------------------------------------
def test_prediction_is_deterministic(trained_model):
    vectorizer, model, _ = trained_model
    text = "Meet me at the library at 5 pm"
    first_run = predict(vectorizer, model, text)
    second_run = predict(vectorizer, model, text)
    assert first_run == second_run, "Model gave different answers for identical input"


# ---------------------------------------------------------------------
# 4. Edge case: empty string must not crash the pipeline.
# ---------------------------------------------------------------------
def test_empty_string_does_not_crash(trained_model):
    vectorizer, model, _ = trained_model
    result = predict(vectorizer, model, "")
    assert result == 0


# ---------------------------------------------------------------------
# 5. Edge case: None input must fail loudly and cleanly, not crash
#    the whole program with an unhandled exception deep in sklearn.
# ---------------------------------------------------------------------
def test_none_input_raises_clean_error(trained_model):
    vectorizer, model, _ = trained_model
    with pytest.raises(ValueError):
        predict(vectorizer, model, None)


# ---------------------------------------------------------------------
# 6. Edge case: wrong type (e.g. an integer) must raise a clear error
#    instead of an obscure sklearn stack trace.
# ---------------------------------------------------------------------
def test_wrong_type_input_raises_clean_error(trained_model):
    vectorizer, model, _ = trained_model
    with pytest.raises(TypeError):
        predict(vectorizer, model, 12345)


# ---------------------------------------------------------------------
# 7. Robustness / invariance test: changing case shouldn't flip the
#    prediction. This is a basic "does the model rely on something it
#    shouldn't" sanity check -- a real ML-testing concept, not just
#    software testing.
# ---------------------------------------------------------------------
def test_prediction_invariant_to_case(trained_model):
    vectorizer, model, _ = trained_model
    lower = predict(vectorizer, model, "win a free prize now")
    upper = predict(vectorizer, model, "WIN A FREE PRIZE NOW")
    assert lower == upper, "Prediction changed when only letter case changed"


# ---------------------------------------------------------------------
# 8. Sanity check: the model shouldn't just predict one class for
#    everything. Catches a degenerate/collapsed model.
# ---------------------------------------------------------------------
def test_model_does_not_collapse_to_single_class(trained_model):
    vectorizer, model, _ = trained_model
    spam_like = predict(vectorizer, model, "Free cash prize click now to claim")
    ham_like = predict(vectorizer, model, "See you at class tomorrow morning")
    assert spam_like != ham_like or (spam_like == 1 and ham_like == 1) is False, (
        "Model may have collapsed to always predicting the same class"
    )
