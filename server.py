import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from joblib import load
from transformers import pipeline, AutoModelForTokenClassification
from transformers.pipelines import PIPELINE_REGISTRY
from pipeline import NER_Pipeline
import datetime
import logging

pd.set_option('display.max_colwidth', 1000)

# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "NER_NLP_tagger",
    pipeline_class=NER_Pipeline,
    pt_model=AutoModelForTokenClassification
)

# load the ner tagger pipeline
ner_tagger = pipeline(
    "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-xlnet-ner-tagging")


def requestResults(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    return output


# setup the flask app
app = Flask(__name__)


def log(message, level=logging.INFO):
    """
    function to log a message
    inputs:
    - message - the message to log
    - level - the logging level of the message
    """
    current_datetime = datetime.datetime.now()
    app.logger.log(level, f"{current_datetime}: {message}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        input = request.form['user-input']
        print(f"input: {input}")
        return redirect(url_for('success', input=input))


@app.route('/success/<input>')
def success(input):
    return "<xmp>" + str(requestResults(input)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)
