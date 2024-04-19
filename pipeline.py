from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline, Pipeline, AutoTokenizer
import torch
import nltk
from pre_processing import PreProcessInput
import spacy


class NER_Pipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        if "model_checkpoint" in kwargs:
            preprocess_kwargs["model_checkpoint"] = kwargs["model_checkpoint"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, model_checkpoint="bert-base-uncased"):
        # get the pos tags for the text
        nlp = spacy.load("en_core_web_sm")
        tokens = []  # make a list for tokens
        pos_tags = []  # make a list of pos_tags
        # perform tokenization
        tokenized = nlp(text)
        for token in tokenized:
            tokens.append(str(token))
            pos_tags.append(str(token.pos_))

        # pre-process the data
        processed_data = PreProcessInput.pre_process_data([tokens], [pos_tags])

        # apply the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, add_prefix_space=True)
        tokenized_inputs = self.tokenizer(
            processed_data, truncation=True, is_split_into_words=True, return_tensors="pt")

        return tokenized_inputs

    def _forward(self, model_inputs):
        self.model_inputs = model_inputs
        outputs = self.model(model_inputs["input_ids"],
                             attention_mask=model_inputs["attention_mask"])
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs.logits.argmax(-1)
        logits = logits[0].tolist()
        logits = logits[1:-1]

        label_list = ['B-O', 'B-AC', 'B-LF', 'I-LF']
        ner_tags = [label_list[i] for i in logits]

        all_tokens = []
        for token_id in self.model_inputs["input_ids"][0]:
            token = self.tokenizer.convert_ids_to_tokens(
                [token_id], skip_special_tokens=False)[0]
            all_tokens.append(token)
        all_tokens = all_tokens[1:-1]

        output = list(zip(all_tokens, ner_tags))

        return output
