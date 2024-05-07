from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline, Pipeline, AutoTokenizer
from huggingface_hub import get_paths_info
import torch
import nltk
from pre_processing import PreProcessInput
import spacy


class NER_Pipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(NER_Pipeline, self).__init__(*args, **kwargs)
        self.model_name = kwargs["tokenizer"].name_or_path
        self.MODEL_SHA = get_paths_info(
            self.model_name, ["model.safetensors", "en"], repo_type="model")[0].lfs.sha256

        # get the device
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        if "model_checkpoint" in kwargs:
            preprocess_kwargs["model_checkpoint"] = kwargs["model_checkpoint"]
        return preprocess_kwargs, {}, {}

    def requires_update(self):
        new_sha = get_paths_info(
            self.model_name, ["model.safetensors", "en"], repo_type="model")[0].lfs.sha256
        if (self.MODEL_SHA == new_sha):
            return False
        else:
            return True

    def preprocess(self, text, model_checkpoint="roberta-base"):
        # get the pos tags for the text
        nlp = spacy.load("en_core_web_sm")
        tokens = []  # make a list for tokens
        pos_tags = []  # make a list of pos_tags
        # perform tokenization
        tokenized = nlp(text)
        for token in tokenized:
            tokens.append(str(token))
            pos_tags.append(str(token.pos_))

        self.inputs = tokens

        # pre-process the data
        processed_data = PreProcessInput.pre_process_data([tokens], [pos_tags])

        # apply the BERT tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_checkpoint, add_prefix_space=True)
        tokenized_inputs = self.tokenizer(
            processed_data, truncation=True, is_split_into_words=True)

        self.tokenized_inputs = tokenized_inputs

        return tokenized_inputs

    def _forward(self, model_inputs):
        self.model_inputs = model_inputs
        self.model.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(model_inputs["input_ids"]).to(self.DEVICE),
                                 attention_mask=torch.tensor(model_inputs["attention_mask"]).to(self.DEVICE))
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs.logits.argmax(-1)
        logits = logits[0].cpu().tolist()
        logits = logits[1:-1]

        label_list = ['B-O', 'B-AC', 'B-LF', 'I-LF']
        ner_tags = [label_list[i] for i in logits]
        processed_ner_tags = [""] * len(self.inputs)

        for i, tag in enumerate(ner_tags):
            try:
                original_token_index = self.tokenized_inputs.token_to_word(
                    i + 1)
                if not original_token_index == None:
                    processed_ner_tags[original_token_index] = tag
            except:
                pass

        # all_tokens = []
        # for token_id in self.model_inputs["input_ids"][0]:
        #     token = self.tokenizer.convert_ids_to_tokens(
        #         [token_id], skip_special_tokens=False)[0]
        #     all_tokens.append(token)
        # all_tokens = all_tokens[1:-1]

        output = list(zip(self.inputs, processed_ner_tags))

        return output
