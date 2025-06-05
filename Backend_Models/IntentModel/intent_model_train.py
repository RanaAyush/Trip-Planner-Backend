from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# Load the dataset
dataset_path = "/mnt/data/intent_classification_large_dataset.json"
with open(dataset_path, "r") as f:
    data = json.load(f)

# Extract texts and labels
texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save the model using joblib
import joblib
model_path = "/mnt/data/intent_classifier_model.joblib"
joblib.dump(pipeline, model_path)

# Return evaluation report and model path
model_path, report
