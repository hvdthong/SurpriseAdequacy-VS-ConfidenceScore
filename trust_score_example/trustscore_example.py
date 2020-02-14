import numpy as np
import matplotlib.pyplot as plt
import trustscore
import trustscore_evaluation

from sklearn import datasets

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

print(X_iris.shape, y_iris.shape)
print(X_digits.shape, y_digits.shape)

datasets = [(X_iris, y_iris), (X_digits, y_digits)]
dataset_names = ["Iris", "Digits"]

from sklearn.linear_model import LogisticRegression

# Train logistic regression on digits.
model = LogisticRegression()
model.fit(X_digits[:1300], y_digits[:1300])

# Get outputs on testing set.
y_pred = model.predict(X_digits[1300:])
# Initialize trust score.
trust_model = trustscore.TrustScore()
trust_model.fit(X_digits[:1300], y_digits[:1300])
# Compute trusts score, given (unlabeled) testing examples and (hard) model predictions.
trust_score = trust_model.get_score(X_digits[1300:], y_pred)

print(trust_score.shape)