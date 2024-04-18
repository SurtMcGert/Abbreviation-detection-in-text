import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from joblib import load

pd.set_option('display.max_colwidth', 1000)

pipeline = load("text_classification.joblib")


def requestResults(kw):
    data = ""
    # pipeline.predict()
    return data


# Boilerplate code from NLP-2023
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        kw = request.form['search']
        return redirect(url_for('success', kw=kw))


@app.route('/success/<kw>')
def success(kw):
    return "<xmp>" + str(requestResults(kw)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)
