# ML Model Test Automation Framework (SMS Spam Classifier)

## Project Overview
Most student ML projects stop at "the model trains and gets X% accuracy."
This project treats a trained machine learning model as a **system under
test** and builds an automated test suite (`pytest`) around it -- the same
way a QA engineer would test any production service, plus a few checks
that are specific to ML systems.

The classifier itself is a small Naive Bayes spam/ham text classifier
(scikit-learn). It is intentionally simple -- the point of this project
is the **testing methodology**, not the model architecture.

## Why This Matters
Testing a machine learning system is not the same as testing regular
software:
- You can't assert exact outputs for every input (models are probabilistic)
- A model can quietly regress in accuracy without ever "crashing"
- Models can behave inconsistently on inputs they weren't tuned for
  (case changes, empty strings, malformed types)
- A model can silently collapse to always predicting one class and still
  "run successfully"

This project's test suite specifically checks for all of the above.

## Test Coverage (8 tests, all passing)
| # | Test | What it catches |
|---|------|------------------|
| 1 | Accuracy threshold | Silent model regression on retrain |
| 2 | Valid output label | Corrupted preprocessing/output pipeline |
| 3 | Deterministic output | Missing random seed / non-reproducible bugs |
| 4 | Empty string handling | Crash on blank input |
| 5 | `None` input handling | Unhandled exception deep in sklearn |
| 6 | Wrong type handling | Unclear stack traces on bad input |
| 7 | Case invariance | Model relying on superficial text features |
| 8 | Class-collapse check | Degenerate model always predicting one class |

## Results
- Holdout set accuracy: **90%**
- Test suite: **8/8 passed**

## How to Run
```bash
pip install -r requirements.txt
python spam_classifier.py          # trains and shows a sample prediction
pytest test_spam_classifier.py -v  # runs the full test suite
```

## Files
- `spam_classifier.py` — model training and prediction logic
- `test_spam_classifier.py` — automated test suite
- `README.md` — this file

## Possible Extensions
- Swap the embedded dataset for the full SMS Spam Collection dataset (Kaggle/UCI)
- Add a fairness/robustness test for adversarial inputs (typos, extra punctuation)
- Wrap in a CI pipeline (GitHub Actions) so tests run automatically on every push
