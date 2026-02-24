import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ========== Step 1: Load Cleaned Review Data ==========
df = pd.read_csv("cleaned_reviews.csv")

# Remove empty reviews (if any)
df = df.dropna(subset=["cleaned_review"])

# ========== Step 2: Build NRC Emotion Lexicon ==========
nrc_path = r"D:\PythonProjects\comment_cleaning\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
            'sadness', 'surprise', 'trust', 'positive', 'negative']

emotion_dict = defaultdict(lambda: {emo: 0 for emo in emotions})

with open(nrc_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            word, emotion, value = line.strip().split('\t')
            if int(value) == 1:
                emotion_dict[word][emotion] = 1
        except ValueError:
            continue  # Skip malformed lines


# ========== Step 3: Define Labeling Function ==========
def get_emotion_vector(text):
    if not isinstance(text, str):
        return [0] * len(emotions)

    vector = [0] * len(emotions)
    for word in text.split():
        if word in emotion_dict:
            for i, emo in enumerate(emotions):
                vector[i] += emotion_dict[word][emo]
    return vector


# Generate emotion vectors
emotion_vectors = df["cleaned_review"].apply(get_emotion_vector)
emotion_df = pd.DataFrame(emotion_vectors.tolist(), columns=emotions)

# Merge with original data
df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)

# ========== Step 4: TF-IDF Weighting + L2 Normalization ==========
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["cleaned_review"])

# Create word emotion matrix
nrc_raw = pd.read_csv(nrc_path, sep='\t', header=None, names=['word', 'emotion', 'value'])
nrc_pivot = nrc_raw[nrc_raw["value"] == 1].pivot(index='word', columns='emotion', values='value').fillna(0)
nrc_pivot = nrc_pivot.reindex(columns=emotions, fill_value=0)

# Create TF-IDF × Emotion matrix
feature_names = vectorizer.get_feature_names_out()
emotion_matrix = np.zeros((len(feature_names), len(emotions)))

for i, word in enumerate(feature_names):
    if word in nrc_pivot.index:
        emotion_matrix[i] = nrc_pivot.loc[word].values

weighted_emotions = tfidf_matrix @ emotion_matrix
weighted_emotions_norm = normalize(weighted_emotions, norm='l2')

# Merge weighted emotions
emotion_weighted_df = pd.DataFrame(weighted_emotions_norm, columns=[f"{emo}_tfidf" for emo in emotions])
df = pd.concat([df.reset_index(drop=True), emotion_weighted_df], axis=1)

# ========== Step 5: NRC × TF-IDF Weighted Scoring ==========
for emo in emotions:
    tfidf_col = f"{emo}_tfidf"
    weighted_col = f"{emo}_weighted"
    df[weighted_col] = df[emo] * df[tfidf_col]

# ========== Step 6: Save Results ==========
df.to_csv("nrc_emotion_features.csv", index=False)
print("✅ Emotion feature processing completed, saved as nrc_emotion_features.csv")

