import pandas as pd
from classify import LLMReviewClassifier
from preprocess import load_and_preprocess_dataset
from feature_engineer import FeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
df = load_and_preprocess_dataset('./data/google_maps_restaurant_reviews/reviews.csv')

if df is not None:
    # 2. Classify reviews for policy violations (Ground Truth)
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
    
    # --- Comparison ---
    print("\n--- Comparing Qwen Ground Truth with Feature Engineering ---")
    
    # Define a simple rule-based classifier based on engineered features
    def rule_based_classifier(row):
        # Rule for advertisement: high sentiment, has URL, low first-person language
        is_ad = (row['sentiment_compound'] > 0.8 and 
                 row['has_url'] and 
                 row['first_person_ratio'] < 0.05)
        
        # Rule for irrelevance: low rating but positive sentiment (mismatch)
        is_irrelevant = (row['rating'] <= 2 and row['sentiment_pos'] > 0.5)
        
        # Rule for rant without visit: very high negativity, high uppercase, many exclamation marks
        is_rant = (row['sentiment_neg'] > 0.8 and 
                   row['uppercase_ratio'] > 0.1 and 
                   row['exclamation_count'] > 3)
        
        return pd.Series([is_ad, is_irrelevant, is_rant], index=['feature_is_ad', 'feature_is_irrelevant', 'feature_is_rant'])

    # Apply the rule-based classifier
    feature_predictions = final_df.apply(rule_based_classifier, axis=1)
    
    # Combine with the final dataframe
    comparison_df = pd.concat([final_df, feature_predictions], axis=1)
    
    # Compare 'is_advertisement'
    print("\n--- Comparison for 'is_advertisement' ---")
    y_true_ad = comparison_df['is_advertisement']
    y_pred_ad = comparison_df['feature_is_ad']
    
    print(classification_report(y_true_ad, y_pred_ad))
    cm_ad = confusion_matrix(y_true_ad, y_pred_ad)
    sns.heatmap(cm_ad, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Advertisement Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Qwen)')
    plt.show()

    print("\n--- Comparison for 'is_irrelevant' ---")
    y_true_irrelevant = comparison_df['is_irrelevant']
    y_pred_irrelevant = comparison_df['feature_is_irrelevant']

    print(classification_report(y_true_irrelevant, y_pred_irrelevant))
    cm_irrelevant = confusion_matrix(y_true_irrelevant, y_pred_irrelevant)
    sns.heatmap(cm_irrelevant, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Irrelevant Content Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Qwen)')
    plt.show()

    print("\n--- Comparison for 'is_rant_without_visit' ---")
    y_true_rant = comparison_df['is_rant_without_visit']
    y_pred_rant = comparison_df['feature_is_rant']

    print(classification_report(y_true_rant, y_pred_rant))
    cm_rant = confusion_matrix(y_true_rant, y_pred_rant)
    sns.heatmap(cm_rant, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Rant Without Visit Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Qwen)')
    plt.show()

    print("\n Comparison complete.")