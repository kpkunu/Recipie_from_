import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("C:/Users/kamal/recipes/ALL-NEW-Recepies.csv")
data = data.dropna(subset=['Ingredients'])
data['Ingredients'] = data['Ingredients'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
data['Ingredients'] = data['Ingredients'].apply(lambda x: ' '.join(x))

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Ingredients'])

# Compute cosine similarity between all recipes
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create Flask app
app = Flask(__name__)

def get_recipe_recommendations(ingredient_input):
    # Transform the user's input into TF-IDF vector
    input_tfidf = tfidf.transform([ingredient_input])
    
    # Compute cosine similarity between the input and all recipes
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get indices of the most similar recipes
    similar_indices = similarity_scores.argsort()[0][-6:][::-1]
    
    recommended_recipes = []
    for index in similar_indices:
        recommended_recipes.append({
            "name": data['Reciepe Name'].iloc[index],
            "ingredients": data['Ingredients'].iloc[index]
        })
    
    return recommended_recipes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients_input = request.form['ingredients']
        recommendations = get_recipe_recommendations(ingredients_input)
        return render_template('index.html', recommendations=recommendations, ingredients=ingredients_input)
    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
