from flask import Flask, render_template, request
from main_model import get_recommendations  # Importing your model function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    selected_genre = ""
    search_keyword = ""
    top_n = 10
    sort_by = ""
    movies = []

    if request.method == "POST":
        selected_genre = request.form.get("genre", "")
        search_keyword = request.form.get("keyword", "")
        sort_by = request.form.get("sort_by", "")
        try:
            top_n = int(request.form.get("top_n", 10))
        except:
            top_n = 10

        # Call your main model's ranking function
        movies = get_recommendations(
            genre=selected_genre,
            keyword=search_keyword,
            top_n=top_n,
            sort_by=sort_by
        )

        # Reverse rating sort order if user selects it
        if sort_by == "rating":
            movies = sorted(movies, key=lambda x: x.get("avg_rating", 0))

    return render_template(
        "index.html",
        genres=genres,
        selected_genre=selected_genre,
        search_keyword=search_keyword,
        top_n=top_n,
        sort_by=sort_by,
        movies=movies
    )

if __name__ == "__main__":
    app.run(debug=True)