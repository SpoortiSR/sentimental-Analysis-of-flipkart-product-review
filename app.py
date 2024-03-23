from flask import Flask, render_template, request
import re
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        review = request.form.get("text")
        data = [review]  # Pass the actual review text, not the string 'review'

        model = joblib.load("E:\Flask\interfernce\sentimental-analysis\sentimental-analysis.pk1")
        prediction = model.predict(data)

        return render_template("output.html", prediction=prediction)

   


if __name__ == '__main__':
    app.run(debug=True)