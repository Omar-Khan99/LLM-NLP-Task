import pickle

# Load model
with open("Models  Files/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Models  Files/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict(text):
    emb = vectorizer.transform([text]).toarray()
    pred = model.predict(emb)[0]
    return "Positive" if pred == 1 else "Negative"

sentiment = predict('I Love one piece it is great')
print("Prediction:", sentiment)