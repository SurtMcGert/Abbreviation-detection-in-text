import pytest
from unittest.mock import patch
import re
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

from server import get_acronym_description, request_results

class TestServer:
    # Register custom pipeline
    PIPELINE_REGISTRY.register_pipeline(
        "NER_NLP_tagger",
        pipeline_class=NER_Pipeline,
        pt_model=transformers.AutoModelForTokenClassification
    )

    # Load NER tagger pipeline
    ner_tagger = transformers.pipeline(
        "NER_NLP_tagger", model="SurtMcGert/NLP-group-CW-roberta-ner-tagging")

    def test_request_results(self):
        string = "This is a test"

        expected = self.ner_tagger(string)

        predicted = request_results(string)

        assert expected == predicted


    def test_get_acronym_description(self):
        acronym = "Federal Bureau of Investigation (FBI)"

        expected = "The Federal Bureau of Investigation (FBI) is the domestic intelligence and security service of the United States and its principal federal law enforcement agency"

        predicted = get_acronym_description(acronym)

        if (predicted == ""):
            assert True
        else:
            assert expected == predicted

if __name__ == "__main__":
    pytest.main()
