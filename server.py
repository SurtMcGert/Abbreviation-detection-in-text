import datetime
import logging
from logging.config import dictConfig

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for, logging as flog
from joblib import load
from transformers import AutoModelForTokenClassification, pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from pipeline import NER_Pipeline

pd.set_option('display.max_colwidth', 1000)

APP_LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'app_file': {  # Handler for app.log
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'default',
        },
        'io_file': {  # Handler for IO.log
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'IO.log',
            'formatter': 'default',
        }
    },
    'loggers': {
        'app': {  # Root logger
            'handlers': ['app_file'],
            'level': 'INFO',  # Not strictly relevant in this scenario
            'propagate': False
        },
        'io': {  # Root logger
            'handlers': ['io_file'],
            'level': 'INFO',  # Not strictly relevant in this scenario
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['app_file']  # Initially, log to app.log
    }
}


# setup the flask app
app = Flask(__name__)
logging.config.dictConfig(APP_LOGGING_CONFIG)
app.logger.level = logging.INFO
app.logger.propagate = True
app.logger = logging.getLogger("app")


# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "NER_NLP_tagger",
    pipeline_class=NER_Pipeline,
    pt_model=AutoModelForTokenClassification
)

# load the ner tagger pipeline
ner_tagger = pipeline(
    "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-roberta-ner-tagging")


def logIO(message):
    app.logger = logging.getLogger("io")
    app.logger.info(f"{message}")
    app.logger = logging.getLogger("app")


def requestResults(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    logIO(f"model-output: {output}")
    return output


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        input = request.form['user-input']
        logIO(f"user-input: {input}")
        return redirect(url_for('success', input=input))


@app.route('/success/<input>')
def success(input):
    return "<xmp>" + str(requestResults(input)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)
