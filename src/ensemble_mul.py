import argparse
import datetime
import json
import numpy as np
import re
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertForQuestionAnswering, AlbertTokenizerFast, XLNetForQuestionAnswering, XLNetTokenizerFast, RobertaTokenizerFast, RobertaForQuestionAnswering

class SquadDataset(Dataset):

    def __init__(self, input_path, tokenizer_checkpoint):
        """
        input_path: path that contains all the files - contexts, questions, answers, answer spans and question ids
        """
        print(f"Reading in Dataset from {input_path}")
        
        with open(input_path + "/context", encoding='utf-8') as f:
            contexts = f.read().split("\t")
        with open(input_path + "/question", encoding='utf-8') as f:
            questions = f.read().split("\t")
        with open(input_path + "/answer", encoding='utf-8') as f:
            answers = f.read().split("\t")
        with open(input_path + "/answer_span", encoding='utf-8') as f:
            spans = f.read().split("\t")
        with open(input_path + "/question_id", encoding='utf-8') as f:
            qids = f.read().split("\t")

        self.contexts = [ctx.strip() for ctx in contexts]#[:100]
        self.questions = [qn.strip() for qn in questions]#[:100]
        self.answers = [ans.strip() for ans in answers]#[:100]
        self.spans = [span.strip().split() for span in spans]#[:100]
        self.start_indices = [int(x[0]) for x in self.spans]
        self.end_indices = [int(x[1]) for x in self.spans]
        self.qids = [qid.strip() for qid in qids]#[:100]

        # intialise XLNetTokenizerFast for input tokenization
        if tokenizer_checkpoint == "xlnet-base-cased":
            self.tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        elif tokenizer_checkpoint == "albert-base-v2":
            self.tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
        elif tokenizer_checkpoint == "roberta-base":
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.tokenizer.padding_side = "right"

        # extract tokenization outputs
        self.tokenizer_dict = self.tokenize()
        self.sample_mapping, self.offset_mapping = self.preprocess()

        self.input_ids = self.tokenizer_dict["input_ids"]
        self.attention_mask = self.tokenizer_dict["attention_mask"]


    def tokenize(self, max_length=384, doc_stride=128):
        """
        inputs:
        1. max_length: specifies the length of the tokenized text
        2. doc_stride: defines the number of overlapping tokens

        output:
        1. tokenizer_dict, which contains
        - input_ids: list of integer values representing the tokenized text; each integer corresponds to a specific token
        - token_type_ids: to distinguish between question and context
        - attention_mask: a binary mask that tells the model which tokens to mask/not mask
        - sample_mapping: map from a feature to its corresponding example, since one question-context pair might give several features
        - offset_mapping: maps each input id with the corresponding start and end characters in the original text

        Tokenize examples (question-context pairs) with truncation and padding, but keep the overflows using a stride specified by `doc_stride`. 
        When the question-context input exceeds the `max_length`, it will contain more than one feature, and each of these features will have context
        that overlaps a bit with the previous features, and the overlapping is determined by `doc_stride`. This is to ensure that although truncation
        is performed, these overflows will ensure that no answer is missed as long as the answer span is shorter than the length of the overlap.
        """
        print("Performing tokenization on dataset")
        tokenizer_dict = self.tokenizer(
            self.questions,
            self.contexts,
            truncation="only_second",
            padding="max_length",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        return tokenizer_dict

    def preprocess(self):
        """
        This functions is to preprocess the outputs of the tokenizer dictionary.
        Due to the possibility that an example has multiple features, this functions ensure that the start_positions and end_positions are mapped
        correctly
        """
        print("Preprocessing Dataset")

        sample_mapping = self.tokenizer_dict.pop("overflow_to_sample_mapping")
        offset_mapping = self.tokenizer_dict.pop("offset_mapping")

        self.tokenizer_dict["start_positions"] = []
        self.tokenizer_dict["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = self.tokenizer_dict["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            sequence_ids = self.tokenizer_dict.sequence_ids(i)

            sample_index = sample_mapping[i]
            answer = self.answers[sample_index]
            start_char = self.start_indices[sample_index]
            end_char = self.end_indices[sample_index]

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # if answer is out of the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                self.tokenizer_dict["start_positions"].append(cls_index)
                self.tokenizer_dict["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                self.tokenizer_dict["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                self.tokenizer_dict["end_positions"].append(token_end_index + 1)
        return sample_mapping, offset_mapping


    def __len__(self):
        """
        Return the number of features in the data
        """
        return len(self.sample_mapping)

    def __getitem__(self, i):

        og_index = self.sample_mapping[i]

        item_dict = {
            "input_ids": torch.tensor(self.input_ids[i]),
            "attention_mask" : torch.tensor(self.attention_mask[i]),
            "start_positions" : torch.tensor(self.tokenizer_dict["start_positions"][i]),
            "end_positions" : torch.tensor(self.tokenizer_dict["end_positions"][i]),
            "og_indices": og_index,
            "og_contexts": self.contexts[og_index],
            "og_questions": self.questions[og_index],
            "og_answers": self.answers[og_index],
            "og_start_indices": self.start_indices[og_index],
            "og_end_indices": self.end_indices[og_index],
            "offset_mapping": torch.tensor(self.offset_mapping[i]),
            "og_question_ids": self.qids[og_index]

        }
        return item_dict

def test_albert(model, dataset, n_best_size=20, max_answer_length=30, device='cpu'):
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    pred = {}

    print("Making Predictions on Test Dataset")
    with torch.no_grad():
        for data in test_dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start = data["start_positions"].to(device)
            end = data["end_positions"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            offset_mapping = data["offset_mapping"]
            context = data["og_contexts"]
            answer = data["og_answers"]
            question = data["og_questions"]
            qids = data["og_question_ids"]

            for i in range(len(input_ids)):

                start_logits = F.softmax(output.start_logits[i], dim=0).cpu().detach().numpy()
                end_logits = F.softmax(output.end_logits[i], dim=0).cpu().detach().numpy()
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                offsets = offset_mapping[i]
                ctxt = context[i]
                qid = qids[i]
                ans = answer[i]

                start_candidates = {}
                end_candidates = {}

                for start in start_indexes:
                  logits = start_logits[start]
                  start_char = offsets[start][0].item()
                  if start_candidates.get(start_char) == None or start_candidates.get(start_char) < logits:
                    start_candidates[start_char] = logits

                for end in end_indexes:
                  logits = end_logits[end]
                  end_char = offsets[end][1].item()
                  if end_candidates.get(end_char) == None or end_candidates.get(end_char) < logits:
                    end_candidates[end_char] = logits

                pred[(qid, ans, ctxt)] = {"start": start_candidates, "end": end_candidates}

    return pred

def test_xlnet(model, dataset, n_best_size=20, max_answer_length=30, device='cpu'):
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    pred = {}

    print("Making Predictions on Test Dataset")
    with torch.no_grad():
        for data in test_dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start = data["start_positions"].to(device)
            end = data["end_positions"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            offset_mapping = data["offset_mapping"]
            context = data["og_contexts"]
            answer = data["og_answers"]
            question = data["og_questions"]
            qids = data["og_question_ids"]

            for i in range(len(input_ids)):
                start_logits = output.start_top_log_probs[i].cpu().detach().numpy()
                end_logits = output.end_top_log_probs[i].cpu().detach().numpy()
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                start_top_indexes = output.start_top_index[i]
                end_top_indexes = output.end_top_index[i]

                offsets = offset_mapping[i]
                ctxt = context[i]
                qid = qids[i]
                ans = answer[i]

                start_candidates = {}
                end_candidates = {}
                for start in start_indexes:
                  logits = start_logits[start]
                  start_index = start_top_indexes[start]
                  start_char = offsets[start_index][0].item()
                  if start_candidates.get(start_char) == None or start_candidates.get(start_char) < logits:
                    start_candidates[start_char] = logits

                for end in end_indexes:
                  logits = end_logits[end]
                  end_index = end_top_indexes[end]
                  end_char = offsets[end_index][1].item()
                  if end_candidates.get(end_char) == None or end_candidates.get(end_char) < logits:
                    end_candidates[end_char] = logits

                pred[(qid, ans, ctxt)] = {"start": start_candidates, "end": end_candidates}

    return pred

def get_answer(model1_pred, model2_pred):
    assert model1_pred.keys() == model2_pred.keys(), "Predictions are not on the same dataset"
    final_predictions = {}
    questions = model1_pred.keys()
    for question in questions:
        qid, actual_ans, context = question
        common_start_keys = set(model1_pred[question]["start"]).intersection(model2_pred[question]["start"])
        common_end_keys = set(model1_pred[question]["end"]).intersection(model2_pred[question]["end"])
        
        common_start = {}
        common_end = {}
        for start in common_start_keys:
            multiplied_score = model1_pred[question]["start"][start] * model2_pred[question]["start"][start]
            common_start[start] = multiplied_score
        for end in common_end_keys:
            multiplied_score = model1_pred[question]["end"][end] * model2_pred[question]["end"][end]
            common_end[end] = multiplied_score
#         highest_start = max(common_start.items(), key=lambda x:x[1])[0]
#         highest_end = max(common_end.items(), key=lambda x:x[1])[0]
#         highest_score_ans = context[highest_start:highest_end]
#         final_predictions.update({actual_ans:highest_score_ans})

        valid_ans = []
        for start in common_start.keys():
            for end in common_end.keys():
                if (end < start) or (end - start + 1) > max_answer_length:
                    continue
                if start <= end:
                    valid_ans.append({
                        "score":common_start[start] + common_end[end],
                        "text":context[start:end]
                    })
        if valid_ans: final_pred = sorted(valid_ans, key=lambda x: x["score"], reverse=True)[0]["text"]
        else: final_pred = ""
        final_predictions.update({qid:final_pred})
    return final_predictions

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    xlnet_dict = torch.load('./xlnet.pt',map_location=torch.device(device))
    # albert_dict = torch.load('./albert.pt',map_location=torch.device(device))
    roberta_dict = torch.load('./roberta.pt',map_location=torch.device(device))
    # xlnet
    xlnet = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(device)
    xlnet.load_state_dict(xlnet_dict["model_state_dict"])
    # # albert
    # albert = AlbertForQuestionAnswering.from_pretrained("albert-base-v2").to(device)
    # albert.load_state_dict(albert_dict["model_state_dict"])
    # roberta
    roberta = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(device)
    roberta.load_state_dict(roberta_dict["model_state_dict"])

    xlnet_data = SquadDataset("../data/curated/test_data", "xlnet-base-cased")
    # albert_data = SquadDataset("../data/curated/test_data", "albert-base-v2")
    roberta_data = SquadDataset("../data/curated/test_data", "roberta-base")

    n_best_size = 20
    max_answer_length = 50

    start = time.time()
    # albert_pred = test_albert(albert, albert_data, n_best_size, max_answer_length=30, device=device)
    roberta_pred = test_albert(roberta, roberta_data, n_best_size, max_answer_length=30, device=device)
    xlnet_pred = test_xlnet(xlnet, xlnet_data, n_best_size, max_answer_length=30, device=device)
    end = time.time()
    print(f'Test Duration using 2 Models: {(end-start)/60}mins')
    final_prediction = get_answer(roberta_pred, xlnet_pred)

    # write model prediction into json file
    with open("./ensemble_mul_output_roberta_xlnet1.json", 'w') as f:
        json.dump(final_prediction, f)