import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from joblib import load
from transformers import pipeline, AutoModelForTokenClassification
from transformers.pipelines import PIPELINE_REGISTRY
from pipeline import NER_Pipeline

pd.set_option('display.max_colwidth', 1000)

# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "NER_NLP_tagger",
    pipeline_class=NER_Pipeline,
    pt_model=AutoModelForTokenClassification
)

ner_tagger = pipeline(
    "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-xlnet-ner-tagging")


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
