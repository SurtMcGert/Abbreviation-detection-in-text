import datetime
import logging.config

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from transformers import AutoModelForTokenClassification, pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from pipeline import NER_Pipeline

# Truncate data frame at length 1000
pd.set_option('display.max_colwidth', 1000)

# Set up logger
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
        'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
}

def log(message, level=logging.INFO):
    """
    function to log a message
    inputs:
    - message - the message to log
    - level - the logging level of the message
    """
    current_datetime = datetime.datetime.now()
    app.logger.log(level, f"{current_datetime}: {message}")

# Set up Flask app
app = Flask(__name__)

# Create logger
logging.config.dictConfig(LOGGING_CONFIG)
app.logger.handlers = logging.getLogger().handlers


# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "NER_NLP_tagger",
    pipeline_class=NER_Pipeline,
    pt_model=AutoModelForTokenClassification
)

# Load NER tagger pipeline
ner_tagger = pipeline(
    "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-xlnet-ner-tagging")


# Get predictions from pipeline
def requestResults(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    app.logger.info(f"model-output: {output}")
    return output


# Home path
@app.route('/')
def home():
    return render_template('index.html')

# Home path method handling
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        # Retrieve user input
        input = request.form['user-input']
        print(f"input: {input}")

        app.logger.info(f"user-input: {input}")
        # Return `success` rout with user input parameter
        return redirect(url_for('success', input=input))

# `/success/<input>` route handling
@app.route('/success/<input>')
def success(input):
    # Print results as HTML page
    return "<xmp>" + str(requestResults(input)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)
