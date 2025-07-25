from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
books = pd.read_csv("books.csv", low_memory=False)
books.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)

# Combine features
books=books.head(5000)
books["Combined"] = books["Book-Title"] + " " + books["Book-Author"]

# TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["Combined"])
similarity = cosine_similarity(tfidf_matrix)

# Recommend function
def recommend(book_title):
    book_title = book_title.lower()
    matches = books[books["Book-Title"].str.lower() == book_title]
    
    if matches.empty:
        return []
    
    index = matches.index[0]
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommended_books = []
    for i, score in sim_scores:
        recommended_books.append({
            "title": books.iloc[i]["Book-Title"],
            "author": books.iloc[i]["Book-Author"],
            "image": books.iloc[i]["Image-URL-M"]
        })
    return recommended_books

@app.route('/', methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        book_name = request.form.get("book")
        recommendations = recommend(book_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)