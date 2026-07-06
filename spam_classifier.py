"""
spam_classifier.py

A small, self-contained SMS spam/ham text classifier.
Built to be TESTED, not just trained -- see test_spam_classifier.py.

No external dataset download required (keeps the project reproducible
for anyone who clones it -- no broken links, no missing files).
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------
# Small embedded dataset (label, message). 1 = spam, 0 = ham (normal message)
# --------------------------------------------------------------------------
DATA = [
    (1, "WIN a free iPhone now, click this link immediately!!!"),
    (1, "Congratulations! You have won a $1000 Walmart gift card. Claim now"),
    (1, "URGENT: your account has been suspended, verify your password here"),
    (1, "Get rich quick! Invest now and double your money in 24 hours"),
    (1, "Free entry in a weekly prize draw, text WIN to 80085"),
    (1, "You have been selected for a cash reward, click to claim"),
    (1, "Limited time offer, buy one get one free, click now"),
    (1, "Your loan has been approved, send your bank details to proceed"),
    (1, "Hot singles in your area want to chat with you now"),
    (1, "Claim your free vacation package, no strings attached, act fast"),
    (1, "Lowest price guaranteed on medication, order without prescription"),
    (1, "You are pre-approved for a credit card with 0% interest, apply now"),
    (1, "Act now to avoid your account being permanently deleted"),
    (1, "Congratulations winner, you get a free cruise, call this number"),
    (1, "Make money from home, no experience needed, click here"),
    (1, "Your package could not be delivered, click here to reschedule"),
    (1, "Earn $5000 a week working from your phone, join today"),
    (1, "Final notice: pay your outstanding balance now to avoid penalty"),
    (1, "Verify your bank account now or it will be closed within 24 hrs"),
    (1, "You've been chosen for an exclusive investment opportunity"),
    (0, "Hey, are we still meeting for lunch tomorrow?"),
    (0, "Can you send me the notes from today's Distributed Systems class?"),
    (0, "Don't forget to submit the DBMS assignment by Friday"),
    (0, "Mom said dinner will be ready by 8 pm"),
    (0, "Let's catch up this weekend, it's been a while"),
    (0, "The train is running 15 minutes late today"),
    (0, "Happy birthday! Hope you have a great day"),
    (0, "I finished the Pattern Recognition unit notes, sending them now"),
    (0, "Can we reschedule our meeting to 4 pm instead of 3?"),
    (0, "Thanks for helping me with the lab report yesterday"),
    (0, "The professor postponed tomorrow's exam to next week"),
    (0, "Are you coming to the college fest this Saturday?"),
    (0, "I left my charger at your place, can I collect it later?"),
    (0, "The wifi in the hostel is down again, anyone else facing this?"),
    (0, "Let me know once you reach home safely"),
    (0, "Good morning, hope you slept well"),
    (0, "The library closes at 6 pm today due to maintenance"),
    (0, "Please review my resume when you get a chance"),
    (0, "See you at the seminar hall in ten minutes"),
    (0, "Can you share the group project slides before class?"),
]

LABELS = [row[0] for row in DATA]
MESSAGES = [row[1] for row in DATA]


def train_model(random_state=42):
    """
    Trains a Naive Bayes spam classifier on the embedded dataset.
    Returns (vectorizer, model, accuracy_on_holdout_set).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        MESSAGES, LABELS, test_size=0.25, random_state=random_state, stratify=LABELS
    )

    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    return vectorizer, model, accuracy


def predict(vectorizer, model, text):
    """
    Predicts whether a single message is spam (1) or ham (0).
    Handles edge cases defensively instead of crashing the caller.
    """
    if text is None:
        raise ValueError("predict() received None -- input message cannot be None")

    if not isinstance(text, str):
        raise TypeError(f"predict() expected a string, got {type(text).__name__}")

    # Empty or whitespace-only input: classify as ham (0) rather than
    # letting the vectorizer choke on an empty vocabulary match.
    if text.strip() == "":
        return 0

    vec = vectorizer.transform([text])
    return int(model.predict(vec)[0])


if __name__ == "__main__":
    vectorizer, model, accuracy = train_model()
    print(f"Holdout accuracy: {accuracy:.2%}")
    sample = "Congratulations, you have won a free prize, click now!"
    print(f"Sample prediction for spam-like message: {predict(vectorizer, model, sample)}")
