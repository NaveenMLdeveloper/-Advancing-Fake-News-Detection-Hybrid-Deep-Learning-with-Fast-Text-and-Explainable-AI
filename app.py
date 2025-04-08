from flask import Flask, render_template, request
import fasttext

# Load trained FastText model
MODEL_PATH = "fake_news_model_1.bin"

try:
    model = fasttext.load_model("best_fasttext_model.bin")
    print(f"✅ Model Loaded Successfully: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

# Initialize Flask app
app = Flask(__name__)

# Home route with form
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        news_text = request.form.get("news_text")
        
        if news_text:
            # Predict using FastText model
            label, _ = model.predict(news_text)

            # Assign label correctly
            if label[0] == "__label__0":  # Fake News
                prediction = "❌ FAKE NEWS"
            else:  # True News
                prediction = "✅ TRUE NEWS"

    return render_template("index.html", prediction=prediction)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
