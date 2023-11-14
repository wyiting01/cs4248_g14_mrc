'''
- to train xlnet and roberta on 80% of train-v1.1.json
python ensemble_unequal_optuna.py --train --roberta --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../xlnet.json" --roberta_dict "../roberta.json"
python ensemble_unequal_optuna.py --train --xlnet --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../xlnet.json" --roberta_dict "../roberta.json"

- to obtain the possible candidates along their scores for validation data (20% of train-v1.1.json) for optuna trials to determine optimal weights (refer to ensemble_optuna.ipynb)
python ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_val.json" --roberta_dict "../ensemble/roberta_val.json"
python ensemble_unequal_optuna.py --get_candidates --roberta --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_val.json" --roberta_dict "../ensemble/roberta_val.json"

- to obtain the possible candidates along their scores for test data and apply optimal weights to obtain final prediction
python ensemble_unequal_optuna.py --get_candidates --xlnet --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json"
python ensemble_unequal_optuna.py --get_candidates --roberta --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --roberta_path "../model/roberta.pt" --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json"
python ensemble_unequal_optuna.py --test --xlnet_dict "../ensemble/xlnet_test.json" --roberta_dict "../ensemble/roberta_test.json" --output_path "../ensemble/ensemble_optuna_pred.json" --xlnet_weight 0.39 --roberta_weight 0.61
'''

import argparse
import datetime
import json 
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import optuna
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, XLNetForQuestionAnswering, XLNetTokenizerFast
import xlnet
import albert
import roberta

def test_roberta(model, dataset, n_best_size=20, device='cpu'):
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
                  if start_candidates.get(start_char) == None or float(start_candidates.get(start_char)) < logits:
                    start_candidates[start_char] = str(logits)

                for end in end_indexes:
                  logits = end_logits[end]
                  end_char = offsets[end][1].item()
                  if end_candidates.get(end_char) == None or float(end_candidates.get(end_char)) < logits:
                    end_candidates[end_char] = str(logits)

                pred[qid] = {"start": start_candidates, "end": end_candidates, "answers": ans, "context": ctxt}

    return pred

def test_xlnet(model, dataset, n_best_size=20, device='cpu'):
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
                  if start_candidates.get(start_char) == None or float(start_candidates.get(start_char)) < logits:
                    start_candidates[start_char] = str(logits)

                for end in end_indexes:
                  logits = end_logits[end]
                  end_index = end_top_indexes[end]
                  end_char = offsets[end_index][1].item()
                  if end_candidates.get(end_char) == None or float(end_candidates.get(end_char)) < logits:
                    end_candidates[end_char] = str(logits)

                pred[qid] = {"start": start_candidates, "end": end_candidates, "answers": ans, "context": ctxt}

    return pred

def weighting_score(xlnet, roberta, w1, w2):
    wdict  = {}
    for qns_id in xlnet.keys():
        xlnet_ans_start = xlnet.get(qns_id).get('start')
        roberta_ans_start = roberta.get(qns_id).get('start')
        ans = roberta.get(qns_id).get('answers')
        ctxt = roberta.get(qns_id).get('context')

        weighted_ans_start = {}

        for key, val in roberta_ans_start.items(): # start w xlnet as it has less candidates
            #roberta_score = roberta_ans_start.get(key)
            xlnet_score = xlnet_ans_start.get(key)
            if xlnet_score == None:
                weighted_ans_start[key] = w2*float(val)
            else:
                 weighted_ans_start[key] = w1*float(xlnet_score) + w2*float(val)

        xlnet_ans_end = xlnet.get(qns_id).get('end')
        roberta_ans_end = roberta.get(qns_id).get('end')
        
        weighted_ans_end = {}

        for key, val in roberta_ans_end.items(): # start w xlnet as it has less candidates
            #roberta_score = roberta_ans_end.get(key)
            xlnet_score = xlnet_ans_end.get(key)
            if xlnet_score == None:
                weighted_ans_end[key] = w2*float(val)
            else:
                weighted_ans_end[key] = w1*float(xlnet_score) + w2*float(val)
        wdict[qns_id] = {"start": weighted_ans_start, "end": weighted_ans_end, "context": ctxt, "answer": ans}
    return wdict

def post_processing(score_dict, max_answer_length=100):
    pred = {}
    for qid in score_dict.keys():
        ans = score_dict.get(qid).get('answer')
        ctxt = score_dict.get(qid).get('context')
        #print(ctxt)
        valid_answer = {}
        start_indexes = score_dict.get(qid)["start"]
        end_indexes = score_dict.get(qid)["end"]
        for start, s_score in start_indexes.items():
            for end, e_score in end_indexes.items():
                start = int(start)
                end = int(end)
                if end < start or end - start + 1 > max_answer_length:
                    continue
                if start <= end:
                    pred_answer = ctxt[start:end]
                    pred_score = s_score + e_score
                    if valid_answer.get(pred_answer) == None or float(valid_answer.get(pred_answer)) < pred_score:
                        valid_answer.update(
                                {pred_answer : pred_score}
                            )

        valid_answes = dict(sorted(valid_answer.items(), key=lambda x: x[1], reverse=True))
        #print(valid_answer)
        if len(valid_answer) == 0:
            pred[qid] = ""
        else:
            pred[qid] = next(iter(valid_answer))
    return pred


def main(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.train:
        # specify hyperparameters
        num_epoch = 2
        batch_size = 16
        learning_rate = 5e-5

        if args.xlnet:
            print("Initialising XLNet Model")
            xlnet_model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased").to(device)
            xlnet_dataset = xlnet.SquadDataset(args.data_path)

            # train the xlnet model
            xlnet.train(model=xlnet_model, dataset=xlnet_dataset, num_epoch=num_epoch, batch_size=batch_size, learning_rate=learning_rate, device=device, model_path=args.xlnet_path)

        if args.roberta:
            print("Initialising roberta Model")
            roberta_model = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(device)
            roberta_dataset = roberta.SquadDataset(args.data_path)

            # train the roberta model
            roberta.train(model=roberta_model, dataset=roberta_dataset, num_epoch=num_epoch, batch_size=batch_size, learning_rate=learning_rate, device=device, model_path=args.roberta_path)
    
    elif args.get_candidates:
      
        if args.roberta:
        
            roberta_model = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(device)
            roberta_checkpt = torch.load(args.roberta_path)
            roberta_model.load_state_dict(roberta_checkpt['model_state_dict'])
            roberta_validation = roberta.SquadDataset(args.data_path)

            roberta_pred = test_roberta(roberta_model, roberta_validation, n_best_size=20, device=device)

            with open(args.roberta_dict, 'w') as f:
                json.dump(roberta_pred, f)

        if args.xlnet:
            
            xlnet_model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased").to(device)
            xlnet_checkpt = torch.load(args.xlnet_path)
            xlnet_model.load_state_dict(xlnet_checkpt['model_state_dict'])
            xlnet_validation = xlnet.SquadDataset(args.data_path)

            xlnet_pred = test_xlnet(xlnet_model, xlnet_validation, n_best_size=20, device=device)

            with open(args.xlnet_dict, 'w') as f:
                json.dump(xlnet_pred, f)
    
    elif args.test:
        f1 = open(args.xlnet_dict)
        xlnet_pred = json.load(f1)
        f2 = open(args.roberta_dict)
        roberta_pred = json.load(f2)

        print("Doing weighting on xlnet and roberta")
        weighted_score_dict = weighting_score(xlnet_pred, roberta_pred, float(args.xlnet_weight), float(args.roberta_weight))
        final_pred = post_processing(weighted_score_dict)

        with open(args.output_path, 'w') as f:
            json.dump(final_pred, f)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--get_candidates', default=False, action='store_true', help='get candidates for test/val data')
    parser.add_argument('--test', default=False, action='store_true', help='generate final prediction for test data')
    parser.add_argument('--xlnet', default=False, action='store_true', help='get candidates for xlnet')
    parser.add_argument('--roberta', default=False, action='store_true', help='get candidates for roberta')
    parser.add_argument('--data_path', help='path to the dataset file')
    parser.add_argument('--xlnet_path', help='path to save trained xlnet model')
    parser.add_argument('--roberta_path', help='path to save trained roberta model')
    parser.add_argument('--xlnet_dict', help='path to save xlnet pred on val/test data')
    parser.add_argument('--roberta_dict', help='path to save roberta pred on val/test data')
    parser.add_argument('--output_path', help='path to save final prediction for test data')
    parser.add_argument('--roberta_weight', default=0.5, help='roberta weight for ensemble')
    parser.add_argument('--xlnet_weight', default=0.5, help='xlnet weight for ensemble')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
    print("Completed!")
    
