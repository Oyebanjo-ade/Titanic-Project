from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved files
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le_sex = joblib.load("model/le_sex.pkl")
le_embarked = joblib.load("model/le_embarked.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = le_sex.transform([request.form["sex"]])[0]
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = le_embarked.transform([request.form["embarked"]])[0]

        features = np.array([[pclass, sex, age, fare, embarked]])
        features_scaled = scaler.transform(features)

        result = model.predict(features_scaled)[0]
        prediction = "Survived 🎉" if result == 1 else "Did Not Survive ❌"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
