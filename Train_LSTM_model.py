import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import json

# Used GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
VOCAB_SIZE = 25000
EMBED_DIM = 100 # change to 128 for used model by random embedding 
HIDDEN_DIM = 64
MAX_LEN = 250
BATCH_SIZE = 64
EPOCHS = 15

# LSTM Model 
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim): # repalce embedding_matrix with  "vocab_size, embed_dim" to used random embedding 
        super(LSTMClassifier, self).__init__()
        num_embeddings, embed_dim = embedding_matrix.shape
        #self.embedding = nn.Embedding(vocab_size, embed_dim) # use random 
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  #Use GloVe
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)  # ← use self.rnn here too
        out = self.fc(hidden[-1])
        return self.sigmoid(out)


# Load preprocessed data
def load_data(path=None):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    texts = [item['review'] for item in data]
    labels = [1 if item['sentiment'] == 'positive' else 0 for item in data]
    return texts, labels


def load_glove_embeddings(glove_path, tokenizer, embed_dim=EMBED_DIM):
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((VOCAB_SIZE, embed_dim))
    for word, i in tokenizer.word_index.items():
        if i < VOCAB_SIZE:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return torch.tensor(embedding_matrix, dtype=torch.float32)



def prepare_tensors(texts, labels):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.float32), tokenizer

def train(model, loader, optimizer, loss_fn):
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (outputs >= 0.5).int().cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

texts, labels = load_data('Dataset/final_dataset.pkl')
X, y, tokenizer = prepare_tensors(texts, labels) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

embedding_matrix = load_glove_embeddings("glove.6B.100d.txt", tokenizer, embed_dim=EMBED_DIM)
model = LSTMClassifier(embedding_matrix, HIDDEN_DIM).to(device)
#model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device) # to use random embedding
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, loss_fn)
    print(f"Epoch {epoch+1}/{EPOCHS} complete.")

y_true, y_pred = evaluate(model, test_loader)
cm = confusion_matrix(y_true, y_pred)

report = classification_report(y_true, y_pred, output_dict=True)
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

metrics = {
    'accuracy': report['accuracy'],
    'precision': report['1.0']['precision'],
    'recall': report['1.0']['recall'],
    'f1-score': report['1.0']['f1-score'],
    'false_positive_rate': fpr,
    'false_negative_rate': fnr
}

with open("metrics_LSTM_glove_with_prepare_Dataset.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

torch.save(model.state_dict(), "model_glove_with_prepare_weights.pt")

with open("tokenizer_glove_with_prepare.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
