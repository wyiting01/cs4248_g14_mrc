import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetForQuestionAnswering, XLNetTokenizerFast
import argparse
import xlnet
import bert
from bert import *
import sys
from bilstm_bert import *
from collections import Counter

torch.manual_seed(0)

class SquadDataset(Dataset):

    def __init__(self, input_path):
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

        self.contexts = [ctx.strip() for ctx in contexts]
        self.questions = [qn.strip() for qn in questions]
        self.answers = [ans.strip() for ans in answers]
        self.spans = [span.strip().split() for span in spans]
        self.start_indices = [int(x[0]) for x in self.spans]
        self.end_indices = [int(x[1]) for x in self.spans]
        self.qids = [qid.strip() for qid in qids]

        # intialise XLNetTokenizerFast for input tokenization
        self.tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.tokenizer.padding_side = "right"

        # extract tokenization outputs
        self.tokenizer_dict = self.tokenize()
        self.sample_mapping, self.offset_mapping = self.preprocess()

        self.input_ids = self.tokenizer_dict["input_ids"]
        self.token_type_ids = self.tokenizer_dict["token_type_ids"]
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

def predict(dataset, xlnet_model, bilstm_model, n_best_size,device='cpu', max_answer_length = 30):
    print("Beginning Prediction")
    pred_data = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    bert = QA('bert/model')
    
    with torch.no_grad():
        for data in pred_data:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start = data["start_positions"].to(device)
            end = data["end_positions"].to(device)

            offset_mapping = data["offset_mapping"]
            context = data["og_contexts"]
            answer = data["og_answers"]
            question = data["og_questions"]
            qids = data["og_question_ids"]

            output = xlnet_model(input_ids=input_ids, attention_mask=attention_mask)

            for i in range(len(input_ids)):
                print(question[i])
                # BERT MODEL
                bert_pred = bert.predict_full(context[i], question[i])

                # XLNET  Model
                pred = {}
                start_logits = output.start_top_log_probs[i].cpu().detach().numpy()
                end_logits = output.end_top_log_probs[i].cpu().detach().numpy()
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                start_top_indexes = output.start_top_index[i]
                end_top_indexes = output.end_top_index[i]

                offsets = offset_mapping[i]
                ctxt = context[i]
                qid = qids[i]

                valid_answers = []
                for start in start_indexes:
                    for end in end_indexes:
                        start_index = start_top_indexes[start]
                        end_index = end_top_indexes[end]
                        # exclude out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context
                        if (
                            start_index >= len(offsets)
                            or end_index >= len(offsets)
                            or offsets[start_index] is None
                            or offsets[end_index] is None
                        ):
                            continue
                        # exclude answers with lengths < 0 or > max_answer_length
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # for valid answers, we will calculate the score and extract the answers out
                        if start_index <= end_index:
                            start_char = offsets[start_index][0]
                            end_char = offsets[end_index][1]
                            valid_answers.append(
                                {
                                    "score": start_logits[start] + end_logits[end],
                                    "text": ctxt[start_char: end_char]
                                }
                            )

                valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
                # if len(valid_answers) == 0:
                #     pred[qid] = ""
                # else:
                #     pred[qid] = valid_answers[0]['text']

                print('XLNET:')
                print(valid_answers)

                # Predicting based off max number of votes from all models.
                ## Tie breaking condition is taking xlnet scores first (Better accuracy for now)
                valid_answers_form = {pred['text'] : pred['score'] for pred in valid_answers}
                valid_answers = list({pred['text'] for pred in valid_answers})
                

                predictions = []
                for i in range(n_best_size):
                    predictions.append(bert_pred[i]['answer'])
                predictions += valid_answers
                print('all preductions counted')
                predictions = Counter(predictions)
                fin = predictions.most_common()
                max_counts = fin[0][1]

                filtered_predictions = {pred : count for pred, count in predictions.items() if count == max_counts}
                print(filtered_predictions)

                pos = {}

                for pred in filtered_predictions.keys():
                    if pred in valid_answers_form.keys():
                        pos[pred] = valid_answers_form[pred]
                
                if len(pos) == 0:
                    for i in range(len(bert_pred)):
                        if bert_pred[i]['answer'] in filtered_predictions.keys():
                            pos[bert_pred[i]['answer']] = bert_pred[i]['confidence']
                print(max(pos, key=pos.get))
                break
            break

def main(args):
    data = SquadDataset(args.data_path)
    model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(torch.device('cpu'))
    checkpoint = torch.load(args.xlnet_model, map_location=torch.device('cpu'))
    bilstm_model = BERT_BiLSTM(28,256,2,2).to(torch.device('cpu')) # Test inputs for now
    model.load_state_dict(checkpoint["model_state_dict"])
    ans = predict(data, model, bilstm_model ,20)
    return ans
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to the dataset file')
    parser.add_argument('--xlnet_model', help='path to xlnet model')
    # parser.add_argument('--bilstm_model', help='path to bilstm model')
    return parser.parse_args()

if __name__ == "__main__":
    print(sys.path)
    args = get_arguments()
    main(args)
    print("Completed!")