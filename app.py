from flask import Flask, render_template, request
from recommender import hybrid_recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    message = None

    if request.method == "POST":
        movie_name = request.form.get("movie_name")
        recommendations, message = hybrid_recommend(movie_name, top_n=10)

    return render_template("index.html", recommendations=recommendations, message=message)

if __name__ == "__main__":
    app.run(debug=True)
