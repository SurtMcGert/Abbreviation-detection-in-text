import logging.config
# from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
import transformers
from transformers.pipelines import PIPELINE_REGISTRY
import threading
import time
from pipeline import NER_Pipeline
from queue import Queue
import streamlit as st
from annotated_text import annotated_text
from collections import Counter, OrderedDict
import pandas as pd
import altair as alt
import wikipediaapi
import re
import wikipedia

PROCESSING_REQUEST = False
UPDATE_AVAILABLE = False
label_list = ['B-O', 'B-AC', 'B-LF', 'I-LF']

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
model_update_checker_thread.setDaemon(True)

model_update_checker_thread.start()

# Get predictions from pipeline


def requestResults(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    return output


def getAcronymsAndLongforms(inp):
    """
    function to get a list of the acronyms and longforms
    inputs:
    - input - the tokens and their NER tags
    """
    acros = {}
    last_was_acro = False
    last_was_b_lf = False
    last_was_i_lf = False
    long_form_builder = ""
    previous_acro = ""
    for pair in inp:
        if pair[1] == "B-AC":
            last_was_acro = True
            precious_acro = pair[0]
            if long_form_builder == "":
                try:
                    acros[pair[0]]
                except:
                    acros[pair[0]] = "no LF found"
            else:
                acros[pair[0]] = long_form_builder
                long_form_builder = ""
        elif pair[1] == "B-LF":
            if last_was_acro == True or (last_was_acro == False & last_was_i_lf == False & last_was_b_lf == False):
                last_was_acro = False
                last_was_i_lf = False
                last_was_b_lf = True
                long_form_builder = pair[0]
        elif pair[1] == "I-LF":
            if last_was_b_lf == True or last_was_i_lf == True:
                last_was_acro = False
                last_was_b_lf = False
                last_was_i_lf = True
                long_form_builder += " " + pair[0]
        elif pair[1] == "B-O":
            if last_was_i_lf == True:
                last_was_acro = False
                last_was_i_lf = False
                last_was_b_lf = False
                if not previous_acro == "":
                    if acros[previous_acro] == "no LF found":
                        acros[previous_acro] = long_form_builder
                        long_form_builder = ""
        output = {"Acronyms": list(acros.keys()),
                  "Long Forms": list(acros.values())}

    return output


def get_acronym_description(acronym):
    """
    This function attempts to retrieve a short description of an acronym from Wikipedia.

    Args:
        acronym: The acronym for which to get a description (string).

    Returns:
        A string containing the short description retrieved from Wikipedia or None if not found.
    """
    try:
        summary = wikipedia.summary(acronym)
        return re.split(r"[.?!]\s*", summary)[0]
    except:
        return ""


st.title("NLP NER Tagger")

# Input widget for user text
user_input = st.text_input("Enter your text here: ")

# Display entered text
if user_input:
    PROCESSING_REQUEST = True
    logger.info(user_input)
    output = requestResults(user_input)
    logger.info(output)
    # display the output from the model as highlighted text
    st.markdown('#')
    annotated_text(output)

    st.markdown('#')
    # Create two columns
    col1, col2 = st.columns(2)

    # count the number of each of the tags present
    counts = {}
    tag_counts = Counter([tag for _, tag in output])
    for label in label_list:
        try:
            counts[label] = tag_counts[label]
        except:
            counts[label] = 0
    tag_counts = counts

    # Display bar chart
    with col1:
        st.bar_chart(tag_counts)

    # get a list of acronyms and their longforms
    acros = getAcronymsAndLongforms(output)
    descriptions = []
    # get a short description of each
    for lf in acros["Long Forms"]:
        lf = lf[0].upper() + lf[1:].lower()
        description = get_acronym_description(lf)
        descriptions.append(description)

    acros["Descriptions"] = descriptions
    # display table
    with col2:
        if len(acros) == 0:
            st.write("There were no acronyms :(")
        else:
            st.table(acros)
    PROCESSING_REQUEST = False
