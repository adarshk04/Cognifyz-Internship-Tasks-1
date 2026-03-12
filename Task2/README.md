# Task 2: Restaurant Recommendation System

## Objective
The goal of this task was to create a content-based recommendation system that suggests restaurants to users based on their specific preferences, such as preferred cuisines and price ranges.

## Process
1. **Data Preprocessing & Cleaning**:
   - I started by loading the dataset and checking for any missing values in the specific columns needed for recommendations.
   - The `Cuisines` column had a few missing entries. Since we rely heavily on text matching for the recommendation engine, I filled the missing entries with the word `'Unknown'`.

2. **Establishing Recommendation Criteria**:
   - The primary criteria I used for finding matching restaurants was **Cuisine Types**.
   - As a secondary filter to narrow down results, the user's **Price range** (1 to 4) is introduced.
   - If multiple restaurants perfectly matched the cuisine and price criteria, they are further sorted by their **Aggregate rating** so the best ones show up first.

3. **Building the Content-Based Filtering Approach**:
   - I used a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** to transform the `Cuisines` strings into a mathematical matrix. This tells us the relative importance of words describing the cuisines across the dataset.
   - I then applied **Cosine Similarity** (`linear_kernel` from scikit-learn) which computes the mathematical distance (relevance) between the user's requested cuisine text, and the cuisines of every single restaurant in the dataset.
   - A perfectly exact cuisine match gets a `Match Score` of 1.0.

4. **Testing and Evaluating the System**:
   - I created a test suite inside the script simulating two different users to see if the recommendations make sense.
   
   **Test Case 1**:
   - Preferences: `Cuisine: 'Italian, Pizza'` and `Price Range: 3`.
   - Result: The system successfully filtered down to restaurants like *Centro*, *Pizza Di Rocco*, and *PizzaExpress* that explicitly offer exactly 'Italian, Pizza' with a price tier of 3, sorted properly by their high ratings.
   
   **Test Case 2**:
   - Preferences: `Cuisine: 'Japanese, Sushi'` and `Price Range: 4`.
   - Result: The system surfaced luxury Japanese restaurants like *Ooma*, *Nobu*, and *MEGU* with absolute precision.

The recommendations returned are highly contextual, matching the exact text representation, maintaining budget restraints, and surfacing the highest quality venues first.

## Files
- `restaurant_recommender.py`: The python script containing the TF-IDF engine and recommendation logic.
- `Dataset .csv`: The local dataset being analyzed.
- `output.txt`: The system output showing the top 5 restaurant recommendations for the test users.
- `README.md`: This markdown documentation detailing the implementation.
