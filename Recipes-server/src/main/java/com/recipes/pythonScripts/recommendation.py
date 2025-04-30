import ast
import pandas as pd
import json
import sys
from ast import literal_eval
from sqlalchemy import create_engine, text as sql_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Database connection configuration
db_config = {
    'user': 'root',
    'password': 'Macair0313',
    'host': 'localhost',
    'database': 'simulaterecipes'
}

# Create database engine using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

# Check if the database connection is successful
try:
    with engine.connect() as connection:
        result = connection.execute(sql_text("SELECT 1"))
        print("Database connection successful:", result.fetchone(), file=sys.stderr)
except Exception as e:
    print(f"Database connection failed: {e}", file=sys.stderr)
    sys.exit(1)

def fetch_data():
    try:
        # Load data from 'Recipes' table
        recipes_query = "SELECT id, title, minutes, nutrition, directions, ingredients FROM Recipes"
        recipes_df = pd.read_sql_query(sql=sql_text(recipes_query), con=engine.connect())
        print("Fetched recipes data", file=sys.stderr)
        print("Recipes DataFrame columns:", recipes_df.columns, file=sys.stderr)

        # Load data from 'Comment' table
        comments_query = "SELECT recipe_id, user_id, rating FROM Comment"
        comments_df = pd.read_sql_query(sql=sql_text(comments_query), con=engine.connect())
        print("Fetched comments data", file=sys.stderr)
        print("Comments DataFrame columns:", comments_df.columns, file=sys.stderr)

        return recipes_df, comments_df
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

def preprocess_data(recipes_df, comments_df):
    try:
        # Rename 'id' column to 'recipe_id'
        recipes_df = recipes_df.rename(columns={'id': 'recipe_id'})

        # Parse 'nutrition' column from JSON
        recipes_df['nutrition'] = recipes_df['nutrition'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        nutrition_columns = ['calories', 'total_fat_PDV', 'sugar_PDV', 'sodium_PDV', 'protein_PDV', 'saturated_fat_PDV', 'carbohydrates_PDV']

        # Expand nutrition JSON to individual columns
        recipes_df[nutrition_columns] = pd.DataFrame(recipes_df['nutrition'].tolist(), index=recipes_df.index)

        # Drop duplicate rows
        recipes_df = recipes_df.drop_duplicates(subset=['recipe_id', 'title', 'minutes', 'directions'])

        # Filter out entries where calories or minutes are zero
        recipes_df = recipes_df[(recipes_df['calories'] != 0) & (recipes_df['minutes'] != 0)]
        recipes_df = recipes_df[(recipes_df[nutrition_columns] != 0).any(axis=1)]

        # Remove outliers
        numerical_columns = recipes_df.select_dtypes(include=['number']).drop(['recipe_id'], axis=1).columns.tolist()
        Q1 = recipes_df[numerical_columns].quantile(0.25)
        Q3 = recipes_df[numerical_columns].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        for col in numerical_columns:
            recipes_df = recipes_df[~(recipes_df[col] > upper_limit[col])]

        # Filter out ratings with value 0
        comments_df = comments_df[comments_df['rating'] != 0]

        # Merge recipes and comments DataFrames
        merged_df = pd.merge(recipes_df, comments_df, on='recipe_id', how='inner')

        # Aggregate rating statistics
        agg_ratings_byrecipe = merged_df.groupby('recipe_id').agg(mean_rating=('rating', 'mean'), number_of_ratings=('rating', 'count')).reset_index()

        # Merge with recipe data
        KB_df = pd.merge(recipes_df, agg_ratings_byrecipe, on='recipe_id', how='inner')

        # Parse 'ingredients' column from JSON
        KB_df['ingredients'] = KB_df['ingredients'].fillna('[]')
        KB_df['ingredients'] = KB_df['ingredients'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        KB_df['ingredients'] = KB_df['ingredients'].apply(lambda x: [ingredient.lower() for ingredient in x] if isinstance(x, list) else [])

        print("Preprocessed data columns:", KB_df.columns, file=sys.stderr)
        return KB_df, comments_df
    except Exception as e:
        print(f"Error preprocessing data: {e}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

def filter_recipes_by_preferences(KB_df, preferences):
    matching_recipes = KB_df.copy()
    try:
        if 'ingredients' in preferences:
            preferred_ingredients = preferences['ingredients']
            include_all_ingredients = preferences.get('include_all_ingredients', False)
            if include_all_ingredients:
                matching_recipes = matching_recipes[matching_recipes['ingredients'].apply(lambda x: set(preferred_ingredients).issubset(set(x)))]
            else:
                matching_recipes = matching_recipes[matching_recipes['ingredients'].apply(lambda x: any(ingredient in x for ingredient in preferred_ingredients))]
        if 'max_time' in preferences:
            max_time = preferences['max_time']
            matching_recipes = matching_recipes[matching_recipes['minutes'] <= max_time]
        if 'max_cals' in preferences:
            max_cals = preferences['max_cals']
            matching_recipes = matching_recipes[matching_recipes['calories'] <= max_cals]
        return matching_recipes
    except Exception as e:
        print(f"Error filtering recipes: {e}", file=sys.stderr)
        return pd.DataFrame()

def calculate_and_sort_scores(KB_df, preference, nutritional_preferences=None):
    try:
        C = KB_df['mean_rating'].mean()
        m = KB_df['number_of_ratings'].quantile(0.9)
        q_recipes = KB_df.loc[KB_df['number_of_ratings'] >= m]
        if not q_recipes.empty:
            q_recipes = q_recipes.copy()
            q_recipes['score'] = q_recipes.apply(lambda x: calculate_score(x, C, m, preference, nutritional_preferences), axis=1)
            q_recipes = q_recipes.sort_values('score', ascending=False)
            return q_recipes
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error calculating and sorting scores: {e}", file=sys.stderr)
        return pd.DataFrame()

def calculate_score(recipe, C, m, preference, nutritional_preferences=None):
    try:
        score = 0  # Score accumulator

        if preference == '4':
            score = (recipe['number_of_ratings'] / (recipe['number_of_ratings'] + m) * recipe['mean_rating']) + \
                    (m / (m + recipe['number_of_ratings']) * C)
        elif preference == '5':
            score = -recipe['calories'] + \
                    (recipe['number_of_ratings'] / (recipe['number_of_ratings'] + m) * recipe['mean_rating']) + \
                    (m / (m + recipe['number_of_ratings']) * C)
        elif preference == '6':
            score = recipe['calories'] + \
                    (recipe['number_of_ratings'] / (recipe['number_of_ratings'] + m) * recipe['mean_rating']) + \
                    (m / (m + recipe['number_of_ratings']) * C)
        elif preference == '7' and nutritional_preferences:
            for pref in nutritional_preferences:
                nutrient, condition = [part.strip() for part in pref.split('(')]
                condition = condition[:-1]
                column_name = f"{nutrient}_PDV"
                if column_name in recipe.index:
                    if condition == 'low':
                        score += -recipe[column_name] + \
                                 (recipe['number_of_ratings'] / (recipe['number_of_ratings'] + m) * recipe['mean_rating']) + \
                                 (m / (m + recipe['number_of_ratings']) * C)
                    elif condition == 'high':
                        score += recipe[column_name] + \
                                 (recipe['number_of_ratings'] / (recipe['number_of_ratings'] + m) * recipe['mean_rating']) + \
                                 (m / (m + recipe['number_of_ratings']) * C)

        # Debug output
        print(f"Recipe ID: {recipe['recipe_id']}, Score: {score}, Number of Ratings: {recipe['number_of_ratings']}, Mean Rating: {recipe['mean_rating']}, Calories: {recipe['calories']}, m: {m}, C: {C}", file=sys.stderr)

        return score
    except Exception as e:
        print(f"Error calculating score: {e}", file=sys.stderr)
        return 0

def collaborative_filtering(comments_df):
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(comments_df[['user_id', 'recipe_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2)
        algo = SVD()
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions)
        return algo
    except Exception as e:
        print(f"Error in collaborative filtering: {e}", file=sys.stderr)
        return None

def content_based_filtering(KB_df, recipe_id, num_recs):
    try:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(KB_df['directions'].apply(lambda x: ' '.join(ast.literal_eval(x))))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(KB_df.index, index=KB_df['recipe_id']).drop_duplicates()
        idx = indices[recipe_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recs + 1]
        recipe_indices = [i[0] for i in sim_scores]
        return KB_df.iloc[recipe_indices]
    except Exception as e:
        print(f"Error in content-based filtering: {e}", file=sys.stderr)
        return pd.DataFrame()

def recommend(user_preferences):
    try:
        recipes_df, comments_df = fetch_data()
        KB_df, comments_df = preprocess_data(recipes_df, comments_df)
        matching_recipes = filter_recipes_by_preferences(KB_df, user_preferences)
        preference = user_preferences.get('preference')
        nutritional_preferences = user_preferences.get('nutritional_preferences', None)
        recommendations = calculate_and_sort_scores(matching_recipes, preference, nutritional_preferences)
        if recommendations.empty:
            recipe_id = recipes_df['recipe_id'].sample().tolist()[0]
            content_recs = content_based_filtering(KB_df, recipe_id, 5)
            result = content_recs['recipe_id'].tolist()
        else:
            result = recommendations['recipe_id'].head(5).tolist()
        return result
    except Exception as e:
        print(f"Error in recommend function: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_preferences = json.loads(sys.argv[1])
        recommendations = recommend(user_preferences)
        print(json.dumps(recommendations))
    else:
        print("Error: Invalid user preferences JSON", file=sys.stderr)
