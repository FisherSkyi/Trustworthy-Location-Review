import pandas as pd
from classify import LLMReviewClassifier
from preprocess import load_and_preprocess_dataset

df = load_and_preprocess_dataset('./data/google_maps_restaurant_reviews/reviews.csv')

classifier = LLMReviewClassifier()
llm_results = classifier.batch_classify(df['review_text'].tolist())
df_llm = pd.DataFrame(llm_results)
print(df_llm.head())