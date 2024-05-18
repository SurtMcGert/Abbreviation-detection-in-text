from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline, Pipeline, AutoTokenizer
from huggingface_hub import get_paths_info
import torch
import nltk
from pre_processing import PreProcessInput
import spacy


class NER_Pipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(NER_Pipeline, self).__init__(*args, **kwargs)
        # save the name of the model
        self.model_name = kwargs["tokenizer"].name_or_path
        self.MODEL_SHA = get_paths_info(
            self.model_name, ["model.safetensors", "en"], repo_type="model")[0].lfs.sha256  # get the sha256 hash of the model

        # get the device
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # set the device

    def _sanitize_parameters(self, **kwargs):
        """
        function to allow for parameters to be passed at any stage of the pipeline process
        """
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        if "model_checkpoint" in kwargs:
            preprocess_kwargs["model_checkpoint"] = kwargs["model_checkpoint"]
        return preprocess_kwargs, {}, {}

    def requires_update(self):
        """
        function to check of the model requires updating
        """
        new_sha = get_paths_info(
            self.model_name, ["model.safetensors", "en"], repo_type="model")[0].lfs.sha256  # get the sha256 hash of the model on the huggingface repo
        # if the new sha is equal to the sha that was loaded when the pipeline was created
        if (self.MODEL_SHA == new_sha):
            return False  # return false
        # otherwise an update is required
        else:
            return True  # return true

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

        # apply the tokenizer
        tokenized_inputs = self.tokenizer(
            processed_data, truncation=True, is_split_into_words=True)

        self.tokenized_inputs = tokenized_inputs  # save the tokenized inputs

        return tokenized_inputs  # return the tokenized inputs

    def _forward(self, model_inputs):
        self.model_inputs = model_inputs  # save the model inputs
        self.model.to(self.DEVICE)  # move the model to the device
        self.model.eval()  # put the model in eval mode
        with torch.no_grad():  # dont calculate gradients
            outputs = self.model(torch.tensor(model_inputs["input_ids"]).to(self.DEVICE),
                                 attention_mask=torch.tensor(model_inputs["attention_mask"]).to(self.DEVICE))  # pass input through model
        return outputs  # return outputs

    def postprocess(self, model_outputs):
        # get prediction for each token
        logits = model_outputs.logits.argmax(-1)
        logits = logits[0].cpu().tolist()  # move output to cpu
        # cut off the first and last elements because they are special tokens
        logits = logits[1:-1]

        label_list = ['B-O', 'B-AC', 'B-LF', 'I-LF']
        # convert the numbers to their actual strings
        ner_tags = [label_list[i] for i in logits]
        # create a list for the final tags
        processed_ner_tags = [""] * len(self.inputs)

        # for each tag predicted
        for i, tag in enumerate(ner_tags):
            try:
                original_token_index = self.tokenized_inputs.token_to_word(
                    i + 1)  # get which input token this tag corresponds to
                # if the index is not None
                if not original_token_index == None:
                    # add this tag to the output list of tags
                    processed_ner_tags[original_token_index] = tag
            except:
                pass

        # pair each input token with its predicted tag
        output = list(zip(self.inputs, processed_ner_tags))

        return output  # return the output
