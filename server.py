import logging.config
# from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
import transformers
from transformers.pipelines import PIPELINE_REGISTRY
import threading
import time
from pipeline import NER_Pipeline
from queue import Queue
import streamlit as st


PROCESSING_REQUEST = False
UPDATE_AVAILABLE = False

APP_LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        },
        'io': {
            'format': '[%(asctime)s]: %(message)s',
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
            'formatter': 'io',
        }
    },
    'loggers': {
        'app': {
            'handlers': ['app_file'],
            'level': 'INFO',  # Not strictly relevant in this scenario
            'propagate': False
        },
        'io': {
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

logging.config.dictConfig(APP_LOGGING_CONFIG)
logger = logging.getLogger("io")


# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "NER_NLP_tagger",
    pipeline_class=NER_Pipeline,
    pt_model=transformers.AutoModelForTokenClassification
)

# Load NER tagger pipeline
ner_tagger = transformers.pipeline(
    "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-roberta-ner-tagging")


def model_update_checker(pipeline):
    global UPDATE_AVAILABLE
    global PROCESSING_REQUEST
    global ner_tagger
    while True:
        while UPDATE_AVAILABLE == False:
            time.sleep(10)
            UPDATE_AVAILABLE = pipeline.requires_update()
        # update model
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            "SurtMcGert/NLP-group-CW-roberta-ner-tagging", force_download=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "SurtMcGert/NLP-group-CW-roberta-ner-tagging")
        UPDATE_AVAILABLE = False
        while PROCESSING_REQUEST == True:
            time.sleep(1)
        ner_tagger = pipeline(
            "NER_NLP_tagger", model=model, tokenizer=tokenizer)


model_update_checker_thread = threading.Thread(
    target=model_update_checker, args=(ner_tagger,))

model_update_checker_thread.start()

# Get predictions from pipeline


def requestResults(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    # logIO(f"model-output: {output}")
    return output


# Title and text display
st.title("Natural Language Processing NER Tagger")

# Input widget for user text
user_input = st.text_input("Enter your text here: ")

# Display entered text
if user_input:
    PROCESSING_REQUEST = True
    logger.info(user_input)
    output = requestResults(user_input)
    logger.info(output)
    st.write(output)
    PROCESSING_REQUEST = False
