import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
import sys
import torch.nn.functional as F
import xlnet

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetForQuestionAnswering, XLNetTokenizerFast

from bert.bert_model import QA
from bilstm_bert import *
from collections import Counter
from xlnet import SquadDataset

'''
Need to install libraries for hyperopt, numpy, sklearn, transformers, BERT:

pip install hyperopt
pip install numpy
pip install torch
pip install -U scikit-learn
pip install transformers
pip install pytorch_transformers

(From main folder cs4248_g14_mrc)

To run this file, use the command:
python src/max_vote.py --data_path data/curated/test_data --xlnet_model model/xlnet.pt --bilstm_model model/bilstm.pt --output_path allMaxVoteAns.json --final_output_path maxVoteAns.json
'''

torch.manual_seed(0)

#### START OF PREDICTION ####
def predict(xlnetDataset, xlnet_model, bilstm_dataset, bilstm_model, n_best_size,device='cpu', max_answer_length = 30):
    print("Beginning Prediction")
    pred_data = DataLoader(dataset=xlnetDataset, batch_size=16, shuffle=False)
    bert = QA('src/bert/model')

    bilstm_pred_data = DataLoader(dataset=bilstm_dataset, batch_size=16, shuffle=False)

    final_pred = {}
    final_pred_all = {}

    correct_pred = 0

    total = len(bilstm_dataset.questions)

    with torch.no_grad():
        print("Running biLSTM")
        quesAns = {}
        for batch in bilstm_pred_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            question_ids = batch.get("question_ids")
            contexts = batch.get("contexts")
            offset_mappings = batch.get("offset_mappings")
            
            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits = start_logits.cpu().detach().numpy() #grad included
            end_logits = end_logits.cpu().detach().numpy()
            
            for i in range(len(input_ids)):
                qid = question_ids[i]
                ctxt = contexts[i]
                start_logit = start_logits[i]
                end_logit = end_logits[i]
                offset = offset_mappings[i]
                answers = []

                for start_index in range(len(start_logit)):
                    for end_index in range(start_index, len(end_logit)):
                        if start_index <= end_index: # for each valid span
                            score = start_logit[start_index] + end_logit[end_index]
                            start_char = offset[start_index][0]
                            end_char = offset[end_index][1]
                            answers.append((score, start_index, end_index, ctxt[start_char:end_char]))

                # sort by top scores
                answers = sorted(answers, key=lambda x: x[0], reverse=True)[:n_best_size]
                pred[qid] = answers[0][3] if answers else ""

                # Save all n_best_size answers' scores
                scores[qid] = [
                    {"score": float(score), "start_logit": float(start_logits[i][start_idx]), "end_logit": float(end_logits[i][end_idx]), "text": text}
                    for score, start_idx, end_idx, text in answers
                ]
                texts = []
                for ans in scores[qid]:
                    texts.append(ans['text'])
                quesAns[qid] = texts
        print("Running XLNet")
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
                if len(valid_answers) == 0:
                    pred[qid] = ""
                else:
                    pred[qid] = valid_answers[0]['text']

                print('XLNET:')
                print(valid_answers)

                # Predicting based off max number of votes from all models.
                ## Tie breaking condition is taking xlnet scores first (Better accuracy for now)
                valid_answers_form = {pred['text'] : pred['score'] for pred in valid_answers}
                valid_answers = list({pred['text'] for pred in valid_answers})
                
                predictions = []
                for i in range(n_best_size):
                    predictions.append(bert_pred[i]['answer'])

                bilstm_preds = quesAns[qid]
                predictions += bilstm_preds

                predictions += valid_answers
                print('all predictions counted')
                predictions = Counter(predictions)
                fin = predictions.most_common()
                max_counts = fin[0][1]

                filtered_predictions = {pred : count for pred, count in predictions.items() if count == max_counts}

                pos = {}

                for pred in filtered_predictions.keys():
                    if pred in valid_answers_form.keys():
                        pos[pred] = valid_answers_form[pred]
                
                if len(pos) == 0:
                    for i in range(len(bert_pred)):
                        if bert_pred[i]['answer'] in filtered_predictions.keys():
                            pos[bert_pred[i]['answer']] = bert_pred[i]['confidence']
                top_answer = max(pos, key=pos.get)

                correct_answer = answer[i]
                # Calculate accuracy
                if top_answer == correct_ans:
                    correct_pred += 1

                # Calculate F1 score
                f1 = calc_f1(top_answer, correct_ans)
                f1_scores.append(f1)

                final_pred[qid] = top_answer
                final_pred_all[qid] = pos
                break
            break

    exact_match = correct_pred / total
    f1 = sum(f1_scores) / total

    print(f"Exact match: {exact_match}")
    print(f"f1: {f1}")
    return final_pred, final_pred_all


def main(args):
    output_path = args.output_path
    final_output_path = args.final_output_path
    xlNetData = SquadDataset(args.data_path)
    xlNetModel = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(torch.device('cpu'))
    checkpoint = torch.load(args.xlnet_model, map_location=torch.device('cpu'))
    xlNetModel.load_state_dict(checkpoint["model_state_dict"])

    bilstm_dataset = biLSTMDataset(in_path=args.data_path)
    bilstm_model = BERT_BiLSTM(64).to(torch.device('cpu'))
    bilstmCheckpoint = torch.load(args.bilstm_model)
    bilstm_model.load_state_dict(bilstmCheckpoint["model_state_dict"])

    final_pred, final_pred_all = predict(xlNetData, xlNetModel, bilstm_dataset, bilstm_model, 20)

    json.dump(final_pred_all, open(output_path,"w"), ensure_ascii=False, indent=4)
    json.dump(final_pred, open(final_output_path, "w"), ensure_ascii=False, indent=4)
    return ans

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='path to the dataset file')
    parser.add_argument('--xlnet_model', required=True, help='path to xlnet model')
    parser.add_argument('--bilstm_model', required=True, help='path to bilstm model')
    parser.add_argument('--output_path', required=True, help='path to all outputs')
    parser.add_argument('--final_output_path', required=True, help= 'path for all single outputs')
    return parser.parse_args()

if __name__ == "__main__":
    #print(sys.path)
    args = get_arguments()
    main(args)
    print("Completed!")