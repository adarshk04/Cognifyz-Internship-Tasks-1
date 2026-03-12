import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data_and_clean(file_path):
    """Loads the restaurant dataset and handles missing values."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Fill missing cuisines with a placeholder so the text vectorizer doesn't complain
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    return df

def setup_content_model(df):
    """Sets up the TF-IDF matrix based on restaurant cuisines."""
    print("Setting up the content-based filtering model...")
    # We use TF-IDF to convert the text of cuisines into a numerical matrix
    # This helps in finding similarities between text strings accurately
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Cuisines'])
    return tfidf, tfidf_matrix

def get_recommendations(pref_cuisine, pref_price, df, tfidf, tfidf_matrix, top_n=5):
    """Returns top N restaurant recommendations based on user preferences."""
    # Step 1: Convert the user's preferred cuisine into the same vector space
    user_tfidf = tfidf.transform([pref_cuisine])
    
    # Step 2: Compute cosine similarity between the user's preference and all restaurants
    # This gives us a score of how closely the cuisines match
    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()
    
    # Create a copy so we don't modify the original dataframe
    results = df.copy()
    results['Match Score'] = cosine_similarities
    
    # Step 3: Filter the results to only include the user's price range budget
    if pref_price is not None:
         results = results[results['Price range'] == pref_price]
    
    # Step 4: Sort restaurants
    # We want the highest match score first. If there's a tie, break it by preferring higher aggregate ratings.
    results = results.sort_values(by=['Match Score', 'Aggregate rating'], ascending=[False, False])
    
    # Filter out anything with a 0 match score (meaning it didn't match the cuisine at all)
    results = results[results['Match Score'] > 0]
    
    # Return the most relevant columns to display
    display_cols = ['Restaurant Name', 'Cuisines', 'City', 'Price range', 'Aggregate rating', 'Match Score']
    return results[display_cols].head(top_n)

if __name__ == "__main__":
    # Ensure correct path to the dataset which is in the parent directory
    dataset_path = 'Dataset .csv'
    
    data = load_data_and_clean(dataset_path)
    vectorizer, matrix = setup_content_model(data)
    
    # --- Testing User 1 ---
    user1_cuisine = "Italian, Pizza"
    user1_price = 3
    print("\n" + "="*50)
    print(f"Evaluating Recommendations for User 1")
    print(f"Preferences -> Cuisine: '{user1_cuisine}', Price Range: {user1_price}")
    print("="*50)
    recs_user1 = get_recommendations(user1_cuisine, user1_price, data, vectorizer, matrix)
    
    if not recs_user1.empty:
        print(recs_user1.to_string(index=False))
    else:
        print("No matches found for these preferences.")
        
    # --- Testing User 2 ---
    user2_cuisine = "Japanese, Sushi"
    user2_price = 4
    print("\n" + "="*50)
    print(f"Evaluating Recommendations for User 2")
    print(f"Preferences -> Cuisine: '{user2_cuisine}', Price Range: {user2_price}")
    print("="*50)
    recs_user2 = get_recommendations(user2_cuisine, user2_price, data, vectorizer, matrix)
    
    if not recs_user2.empty:
        print(recs_user2.to_string(index=False))
    else:
        print("No matches found for these preferences.")
