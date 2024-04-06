from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["get","post"])
def prediction():
    if request.method == "POST":

        input_text =request.form.get("text")

        data_point = [input_text]

        model = joblib.load('best_models/decision_tree.pkl')

        prediction = model.predict(data_point)
  
        
        predicted_label = 'positive' if prediction[0] == 1 else 'negative'
        return render_template('output.html', prediction=predicted_label)



if __name__ =="__main__":
    app.run(debug=True)