pip install spacy
#!pip install black
#pip install click
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
pip install https://confirmit-filex-public.sermo.com/content/2023/ES/en_core_web_sm-3.7.0-py3-none-any.whl

import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load a spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Sample data (replace with your own dataset)
data = {
    'response_id': [1, 2, 3],
    'response_text': [
        "I experienced a severe headache after taking the medication.",
        "I felt fine after using the product.",
        "I had a skin rash and itchiness after using the cream."
    ]
}

# Create a DataFrame from your data
df = pd.DataFrame(data)

# Define adverse event keywords
adverse_keywords = ['severe', 'headache', 'rash', 'itchiness']

# Function to check for potential adverse events in a text
def has_adverse_event(text):
    doc = nlp(text.lower())
    for keyword in adverse_keywords:
        if keyword in doc.text:
            return True
    return False

# Apply the has_adverse_event function to each response
df['potential_adverse_event'] = df['response_text'].apply(has_adverse_event)

# Filter for potential adverse events
potential_adverse_events = df[df['potential_adverse_event']]

# Print potential adverse events
print(potential_adverse_events[['response_id', 'response_text']])

# Optionally, you can perform more advanced analysis like TF-IDF similarity to find similar reports.
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['response_text'])

# Calculate cosine similarity between responses
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print responses with high cosine similarity (indicating similarity in content)
for i in range(len(df)):
    similar_responses = [(j, cosine_sim[i][j]) for j in range(len(df)) if i != j and cosine_sim[i][j] > 0.5]
    if similar_responses:
        print(f"Response {i+1} is similar to:")
        for response_idx, similarity in similar_responses:
            print(f"  - Response {response_idx+1} (Similarity: {similarity:.2f})")
