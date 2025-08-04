import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
#from xgboost import XGBClassifier


# Load prepare data
def load_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    texts = [item['review'] for item in data]
    labels = [1 if item['sentiment'] == 'positive' else 0 for item in data]
    return texts, labels

texts, labels = load_data("Dataset/final_dataset.pkl")

def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1-score': report['1']['f1-score'],
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }

print("Generating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
embeddings = vectorizer.fit_transform(texts).toarray()


print("Training classifier...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
#clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss') # To Train by XGBoost
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

metrics = compute_metrics(y_test, y_pred)
print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")



# Save trained model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save embedding
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save evaluation metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel and metrics saved successfully.")