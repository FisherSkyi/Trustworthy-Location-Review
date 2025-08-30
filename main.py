import pandas as pd
from classify import LLMReviewClassifier
from preprocess import load_and_preprocess_dataset
from feature_engineer import FeatureExtractor

# 1. Load and preprocess the data
df = load_and_preprocess_dataset('./data/google_maps_restaurant_reviews/reviews.csv')

if df is not None:
    # 2. Classify reviews for policy violations
    classifier = LLMReviewClassifier()
    llm_results = classifier.batch_classify(df['review_text'].tolist())
    df_llm = pd.DataFrame(llm_results)
    
    # 3. Engineer features from the text
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_all_features(df)
    
    # 4. Combine all data into a final DataFrame
    final_df = pd.concat([df.reset_index(drop=True), df_llm, features_df], axis=1)
    
    print("\n--- Pipeline Complete ---")
    print("Final DataFrame shape:", final_df.shape)
    print("\nFinal DataFrame head:")
    print(final_df.head())
    
    # You can now save this final_df or use it for modeling
    # final_df.to_csv('./data/processed_reviews_with_features.csv', index=False)
    # print("\n✅ Final DataFrame saved to './data/processed_reviews_with_features.csv'")