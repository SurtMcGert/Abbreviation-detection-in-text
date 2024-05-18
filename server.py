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
    """
    function for the model update thread to run
    inputs:
    pipeline - the models pipeline
    """
    global UPDATE_AVAILABLE
    global PROCESSING_REQUEST
    global ner_tagger
    # loop infinitely
    while True:
        # while there is no update available
        while UPDATE_AVAILABLE == False:
            time.sleep(10)
            # check if the pipeline requires an update
            UPDATE_AVAILABLE = pipeline.requires_update()
        # update model
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            "SurtMcGert/NLP-group-CW-roberta-ner-tagging", force_download=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "SurtMcGert/NLP-group-CW-roberta-ner-tagging")
        # wait for the server to finish processing requests
        while PROCESSING_REQUEST == True:
            time.sleep(1)
        ner_tagger = transformers.pipeline(
            "NER_NLP_tagger", model=model, tokenizer=tokenizer)  # set the new pipeline with the new model and tokenizer
        pipeline = ner_tagger
        UPDATE_AVAILABLE = False  # declare the update finished


model_update_checker_thread = threading.Thread(
    target=model_update_checker, args=(ner_tagger,))  # make a thread to run the model_update_checker function
model_update_checker_thread.setDaemon(True)  # prevent thread hanging

model_update_checker_thread.start()  # begin the thread


def request_results(input):
    """
    function to get result from model
    inputs:
    - input - the text to pass to the model
    """
    output = ner_tagger(input)
    return output


def get_acronyms_and_long_forms(inp):
    """
    function to get a list of the acronyms and longforms
    inputs:
    - input - the tokens and their NER tags
    """
    acros = {}  # a dictionary to store the found acronyms and their long forms
    last_was_acro = False  # was the last token an acronym
    last_was_b_lf = False  # was the last token the beginning of a long form
    last_was_i_lf = False  # was the last token inside a long form
    long_form_builder = ""  # string to build the current long form
    previous_acro = ""  # the last found acronym
    # for each token and NER tag pair in the input
    for pair in inp:
        # if token is an acronym
        if pair[1] == "B-AC":
            last_was_acro = True  # say that the last token was an acronym
            last_was_i_lf = False  # say that the last token was not a long form
            last_was_b_lf = False  # say that the last token was not the start of a long form
            previous_acro = pair[0]  # set the last known acronym to be this
            # if there is no long form currently built
            if long_form_builder == "":
                try:
                    # try to access this acronym to check if it is already stored in the dictionary
                    acros[pair[0]]
                except:
                    # if there is an error (indicating this acronym has not been added to the dictionary), set its long form to "no LF found"
                    acros[pair[0]] = "no LF found"
            else:
                # otherwise there is a currently built long form
                # add this acronym to the dictionary and set its value to the currently built long form
                acros[pair[0]] = long_form_builder
                long_form_builder = ""  # clear the long form
        # else, if the token is the beginning of a long form
        elif pair[1] == "B-LF":
            # if the last token was either an acronym or neither an acronym or a long form
            # if last_was_acro == True or (last_was_acro == False & last_was_i_lf == False & last_was_b_lf == False):
            last_was_acro = False  # say that the last token was not an acronym
            last_was_i_lf = False  # say that the last token was not a long form
            last_was_b_lf = True  # say that the last token was the beginning of a long form
            # start building the long form by adding this token to the start of the long form builder string
            long_form_builder = pair[0]
        # else, if the token is inside a long form
        elif pair[1] == "I-LF":
            # if the last token was either the start of a long form or was inside of a long form
            if last_was_b_lf == True or last_was_i_lf == True:
                # add this token to the long form currently being built
                long_form_builder += " " + pair[0]
            last_was_acro = False  # say that the last token was not an acronym
            last_was_b_lf = False  # say that the last token was not the start of a long form
            last_was_i_lf = True  # say that the last token was inside a long form
        # else, if the token is neither an acronym nor a long form
        elif pair[1] == "B-O":
            # if the last token was inside a long form
            if last_was_i_lf == True:
                # if there is a previous acronym
                if not previous_acro == "":
                    # if that acronym doesnt have a long form
                    if acros[previous_acro] == "no LF found":
                        # set the long form of this acronym to be the long form just built
                        acros[previous_acro] = long_form_builder
                        long_form_builder = ""  # clear the long form
            last_was_acro = False  # say that the last token was not an acronym
            last_was_i_lf = False  # say that the last token was not inside a long form
            last_was_b_lf = False  # say that the last token was not the beginning of a long form
        # output a dictionary of acronyms and long forms
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
        # search for the acronym in wikipedia
        summary = wikipedia.summary(acronym)
        # get the first sentence of the summary
        return re.split(r"[.?!]\s*", summary)[0]
    except:
        return ""


# set the title of the site
st.title("NLP NER Tagger")

# Input widget for user text
user_input = st.text_input("Enter your text here: ")

# Display entered text
if user_input:
    PROCESSING_REQUEST = True
    logger.info(user_input)
    output = request_results(user_input)
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
    acros = get_acronyms_and_long_forms(output)
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
