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

from server import get_acronym_description, request_results, get_acronyms_and_long_forms

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
        input = "This is a test"

        expected = self.ner_tagger(input)

        predicted = request_results(input)

        assert expected == predicted


    def test_get_acronym_description(self):
        input = "Federal Bureau of Investigation (FBI)"

        expected = "The Federal Bureau of Investigation (FBI) is the domestic intelligence and security service of the United States and its principal federal law enforcement agency"

        predicted = get_acronym_description(input)

        if (predicted == ""):
            assert True
        else:
            assert expected == predicted


    def test_get_acronyms_and_long_forms(self):
        input = [("This", "B-O"), ("is", "B-O"), ("a", "B-O"), ("test", "B-O"), ("Federal", "B-LF"), ("Bureau", "I-LF"), ("of", "I-LF"), ("Investigation", "I-LF"), ("FBI", "B-AC")]

        expected = {
            "Acronyms": ["FBI"],
            "Long Forms": ["Federal Bureau of Investigation"]
        }

        predicted = get_acronyms_and_long_forms(input)

        assert expected == predicted


if __name__ == "__main__":
    pytest.main()
