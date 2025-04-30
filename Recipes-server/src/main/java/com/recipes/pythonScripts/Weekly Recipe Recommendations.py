import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text as sql_text
from ast import literal_eval

# Database connection configuration
db_config = {
    'user': 'root',
    'password': 'Macair0313',
    'host': 'localhost',
    'database': 'simulaterecipes'
}
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

# Get user preferences from database
def get_user_preferences(user_id, engine):
    favorites_query = f"SELECT recipe_id FROM Favorite WHERE user_id = {user_id}"
    likes_query = f"SELECT recipe_id FROM `Like` WHERE user_id = {user_id}"
    comments_query = f"SELECT recipe_id, rating FROM Comment WHERE user_id = {user_id}"

    favorite_recipes = pd.read_sql_query(sql_text(favorites_query), con=engine.connect())
    liked_recipes = pd.read_sql_query(sql_text(likes_query), con=engine.connect())
    commented_recipes = pd.read_sql_query(sql_text(comments_query), con=engine.connect())

    return {
        'favorite_recipes': favorite_recipes['recipe_id'].tolist(),
        'liked_recipes': liked_recipes['recipe_id'].tolist(),
        'commented_recipes': commented_recipes[['recipe_id', 'rating']].values.tolist()
    }

# Collaborative filtering using matrix factorization
def collaborative_filtering(comments_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(comments_df[['user_id', 'recipe_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

# Get top-N collaborative filtering recommendations
def get_collaborative_recommendations(user_id, algo, all_recipes, top_n=10):
    user_history = all_recipes[all_recipes['user_id'] == user_id]['recipe_id'].tolist()
    predictions = [algo.predict(user_id, recipe_id) for recipe_id in all_recipes['recipe_id'].unique() if recipe_id not in user_history]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommended_recipes = [pred.iid for pred in predictions[:top_n]]
    return recommended_recipes

# Apply tag-based weighting to recommendations
def tag_weighting(recommendations_df, user_tags):
    for tag, weight in user_tags.items():
        recommendations_df[tag] = recommendations_df[tag].apply(lambda x: x * weight if x else 1)
    recommendations_df['weighted_score'] = recommendations_df.apply(lambda row: row['collaborative_score'] * row[tag], axis=1)
    return recommendations_df.sort_values(by='weighted_score', ascending=False)

# Compute user preference-based score
def calculate_personalized_score(row, user_preferences):
    score = 0
    if row['recipe_id'] in user_preferences['favorite_recipes']:
        score += 10
    if row['recipe_id'] in user_preferences['liked_recipes']:
        score += 5
    return score

# Integrate user preferences into final score
def integrate_user_preferences(recommendations_df, user_preferences):
    recommendations_df['personalized_score'] = recommendations_df.apply(lambda row: calculate_personalized_score(row, user_preferences), axis=1)
    recommendations_df['final_score'] = recommendations_df['personalized_score'] * 0.5 + recommendations_df['weighted_score'] * 0.5
    return recommendations_df.sort_values(by='final_score', ascending=False)

# Content-based filtering based on recipe instructions
def content_based_filtering(KB_df, user_history_recipes, top_n=5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(KB_df['directions'].apply(lambda x: ' '.join(literal_eval(x))))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    recommendations = []
    for recipe_id in user_history_recipes:
        idx = KB_df[KB_df['recipe_id'] == recipe_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommendations.extend([KB_df['recipe_id'].iloc[i[0]] for i in sim_scores[1:top_n+1]])

    return list(set(recommendations))

# Generate a personalized weekly meal plan
def generate_weekly_meal_plan(user_id, user_tags, user_preferences, engine, KB_df, comments_df):
    algo = collaborative_filtering(comments_df)
    collaborative_recs = get_collaborative_recommendations(user_id, algo, KB_df, top_n=50)

    user_history_recipes = [r[0] for r in user_preferences['commented_recipes']]
    content_recs = content_based_filtering(KB_df, user_history_recipes, top_n=10)

    recommendations_df = KB_df[KB_df['recipe_id'].isin(collaborative_recs + content_recs)].copy()
    recommendations_df['collaborative_score'] = recommendations_df['recipe_id'].apply(lambda x: collaborative_recs.count(x))

    recommendations_df = tag_weighting(recommendations_df, user_tags)
    final_recommendations = integrate_user_preferences(recommendations_df, user_preferences)

    weekly_plan = []
    for day in range(7):
        daily_candidates = final_recommendations.sample(n=5)
        daily_plan = daily_candidates.sample(n=1)
        weekly_plan.append(daily_plan)

    return weekly_plan

# Main entry point to run weekly meal plan generation
def main():
    user_id = 1
    user_tags = {
        'High protein': 1.5,
        'Lean and green': 1.2,
        'Quick and easy': 1.3
    }

    # Load recipe and comment data from database
    recipes_query = "SELECT * FROM Recipes"
    comments_query = "SELECT * FROM Comment"

    recipes_df = pd.read_sql_query(sql_text(recipes_query), con=engine.connect())
    comments_df = pd.read_sql_query(sql_text(comments_query), con=engine.connect())

    # Fetch user preferences
    user_preferences = get_user_preferences(user_id, engine)

    # Generate personalized weekly meal plan
    weekly_meal_plan = generate_weekly_meal_plan(user_id, user_tags, user_preferences, engine, recipes_df, comments_df)

    # Output the recommended plan
    for day, plan in enumerate(weekly_meal_plan, 1):
        print(f"Day {day}:")
        print(plan[['recipe_id', 'title']].to_string(index=False))

if __name__ == "__main__":
    main()
