from preprocess import load_and_preprocess_dataset

def main():
    # Load the dataset
    df = load_and_preprocess_dataset('./data/google_maps_restaurant_reviews/reviews.csv')

    if df is not None:
        print("\n Dataset Info:")
        print(df.info())

        print(f"\n Dataset shape: {df.shape}")
        print("\n First 5 reviews:")
        print(df.head())

        # Display data quality info
        print(f"\n Data Quality Check:")
        print(f"- Total reviews: {len(df)}")
        print(f"- Average review length: {df['review_text'].str.len().mean():.1f} characters")

if __name__ == "__main__":
    main()