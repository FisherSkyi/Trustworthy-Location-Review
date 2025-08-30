from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import pandas as pd
from preprocess import load_and_preprocess_dataset
import re
import numpy as np

class FeatureExtractor:
    """Extract comprehensive features from review text and metadata"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def extract_text_features(self, text):
        """Extract textual features from review text"""
        if not isinstance(text, str):
            text = ""
        features = {}

        # Basic text statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0

        # Character-level features
        features['char_count'] = len(text)
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
        features['digit_count'] = sum(1 for c in text if c.isdigit())

        # Ratios
        features['uppercase_ratio'] = features['uppercase_count'] / max(1, features['char_count'])
        features['punctuation_ratio'] = features['punctuation_count'] / max(1, features['char_count'])
        features['digit_ratio'] = features['digit_count'] / max(1, features['char_count'])

        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text)
        features.update({
            'sentiment_pos': sentiment_scores['pos'],
            'sentiment_neu': sentiment_scores['neu'],
            'sentiment_neg': sentiment_scores['neg'],
            'sentiment_compound': sentiment_scores['compound']
        })

        # URL and mention detection
        features['has_url'] = bool(re.search(r'http[s]?://\S+|www\.\w+\.\w+', text.lower()))
        features['has_email'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['has_phone'] = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))

        # Exclamation and question marks
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')

        # First person indicators
        first_person_words = ['i', 'me', 'my', 'myself', 'we', 'us', 'our']
        text_words = text.lower().split()
        features['first_person_count'] = sum(1 for word in text_words if word in first_person_words)
        features['first_person_ratio'] = features['first_person_count'] / max(1, len(text_words))

        return features

    def extract_metadata_features(self, row):
        """Extract features from metadata"""
        features = {}

        # Rating-based features
        features['rating'] = row['rating']
        features['is_extreme_rating'] = 1 if row['rating'] in [1, 5] else 0
        features['is_low_rating'] = 1 if row['rating'] <= 2 else 0
        features['is_high_rating'] = 1 if row['rating'] >= 4 else 0

        # Temporal features (if timestamp is available)
        if 'timestamp' in row and pd.notna(row['timestamp']):
            timestamp = pd.to_datetime(row['timestamp'])
            features['day_of_week'] = timestamp.dayofweek
            features['hour_of_day'] = timestamp.hour
            features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0

        return features

    def extract_all_features(self, df):
        """Extract all features for the entire dataset"""
        all_features = []

        for idx, row in df.iterrows():
            text_features = self.extract_text_features(row['review_text'])
            metadata_features = self.extract_metadata_features(row)

            # Combine all features
            combined_features = {**text_features, **metadata_features}
            all_features.append(combined_features)

        return pd.DataFrame(all_features)

if __name__ == '__main__':
    df = load_and_preprocess_dataset('./data/google_maps_restaurant_reviews/reviews.csv')
    if df is not None:
        # Extract features
        feature_extractor = FeatureExtractor()
        features_df = feature_extractor.extract_all_features(df)

        print("Feature Engineering Complete!")
        print(f"Extracted {len(features_df.columns)} features")
        print("\nFeature columns:")
        print(list(features_df.columns))

        # Combine with original data
        df_with_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        print(f"\nDataset shape with features: {df_with_features.shape}")

        # Display feature statistics
        print("\nFeature Statistics:")
        print(features_df.describe())
