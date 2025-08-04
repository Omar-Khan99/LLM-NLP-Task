import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

data = pd.read_csv("Dataset\IMDB Dataset.csv" , sep = "," , encoding = "utf-8")

# Initialize lemmatizer and stop words set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensure the input is a string
    if not isinstance(text, str):
        return "" 

    # Prepare Data
    text = text.lower()
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"\S*@\S*", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Tokenize text and remove stop words
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmas = [lemmatizer.lemmatize(word) for word in words]

    # Join the words together
    return " ".join(lemmas)

data['review_cleaned'] = data['review'].apply(preprocess_text)

processed_data = []
for _ ,row in data.iterrows():
    processed_data.append({'review': row['review_cleaned'], 'sentiment': row['sentiment']})

with open('final_dataset.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
