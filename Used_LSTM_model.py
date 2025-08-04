import torch
import torch.nn as nn
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
VOCAB_SIZE = 25000
EMBED_DIM = 100 # change to 128 for used model by random embedding 
HIDDEN_DIM = 64
MAX_LEN = 250

# used GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model 
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim): # repalce embedding_matrix with  "vocab_size, embed_dim" to used random embedding 
        super(LSTMClassifier, self).__init__()
        num_embeddings, embed_dim = embedding_matrix.shape
        #self.embedding = nn.Embedding(vocab_size, embed_dim) # use random 
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  # use GloVe
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
    

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

with open("Models  Files/tokenizer_glove_with_prepare.pkl", "rb") as f: # if change below model change tokenizer with same name of model
    tokenizer = pickle.load(f)


embedding_matrix = load_glove_embeddings("glove.6B.100d.txt", tokenizer, embed_dim=EMBED_DIM)
model = LSTMClassifier(embedding_matrix, HIDDEN_DIM).to(device)
#model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM) # use this if load any model not used GloVe
model.load_state_dict(torch.load("Models  Files/model_glove_with_prepare_weights.pt", map_location=device)) # to used any model replace name 
model.to(device)
model.eval()

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        prediction = (output >= 0.5).int().item()

    label = "Positive" if prediction == 1 else "Negative"
    return label

print(predict_sentiment("I watched one piece and i Love it."))

