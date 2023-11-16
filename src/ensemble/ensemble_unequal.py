'''
- to obtain results of the ensemble method with unequal probabiltiy weighting
python3 src/ensemble_unequal.py --test --xlnet_dict src/xlnet_val.json --roberta_dict src/roberta_val.json --xlnet_acc src/xlnet_kf_scores.json --roberta_acc src/roberta_kf_scores.json --output_path src 

'''

import argparse
import datetime
import json 
import numpy as np
import re
import string
import collections
import math
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, XLNetForQuestionAnswering, XLNetTokenizerFast
import xlnet
import albert
import roberta

def calc_autotune_alpha(accs: list) -> int:
    '''
    Calculates appropriate alpha for the autotuning weighting method,
    which is taking the floor of the root difference between the 2 models with largest accuracies.
    '''
    if len(accs) < 2:
        raise ValueError("The list of accuracies must contain at least two elements.")

    largest, second_largest = heapq.nlargest(2, accs)
    diff = largest - second_largest
    root_diff = math.sqrt(diff)
    alpha = math.floor(root_diff)

    return alpha

def aggregate_predictions(all_preds: dict, alpha: int, accs: list) -> dict:
    """ Aggregate predictions from all models. """

    weights = [acc**alpha for acc in accs]

    aggregated_preds = {}
    for qid, preds in all_preds[0].items():  # Initialize strcuture fst
        aggregated_preds[qid] = {pos: {} for pos in preds.keys()}

    for i, model_preds in enumerate(all_preds):
        weight = weights[i]
        for qid, pos in model_preds.items():
            for pos_type, pos_preds in pos.items(): # start, {scores}
                if pos_type == "answers" or pos_type == "context":
                    continue
                for char_pos, prob in pos_preds.items(): # char pos, prob
                    if char_pos in aggregated_preds[qid][pos_type]:
                        aggregated_preds[qid][pos_type][char_pos] += float(prob) * weight
                    else:
                        aggregated_preds[qid][pos_type][char_pos] = float(prob) * weight

    # Normalize Ps and Pe for each char position
    for qid, pos in aggregated_preds.items():
        for pos_type in pos.keys():
            if pos_type == "answers" or pos_type == "context":
                continue
            total_prob = sum(float(value) for value in aggregated_preds[qid][pos_type].values())
            for pos in aggregated_preds[qid][pos_type]:
                aggregated_preds[qid][pos_type][pos] /= total_prob
            aggregated_preds[qid]['answers'] = all_preds[0][qid]['answers']
            aggregated_preds[qid]['context'] = all_preds[0][qid]['context']

    return aggregated_preds

def post_processing(score_dict, max_answer_length=100):
    pred = {}
    for qns in score_dict.keys():
        ans = score_dict.get(qns).get('answer')
        ctxt = score_dict.get(qns).get('context')
        valid_answer = {}
        start_indexes = score_dict.get(qns)["start"]
        end_indexes = score_dict.get(qns)["end"]
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

        valid_answer = dict(sorted(valid_answer.items(), key=lambda x: x[1], reverse=True))
        #print(valid_answer)
        if len(valid_answer) == 0:
            print(qns, ans, valid_answer)
            pred[qns] = (" ", ans)
            
        else:
            pred[qns] = (next(iter(valid_answer)), ans)
    return pred

def main(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.test:
        output_path = args.output_path

        # all models 
        f1 = open(args.xlnet_dict)
        xlnet_pred = json.load(f1)
        f2 = open(args.roberta_dict)
        roberta_pred = json.load(f2)

        all_preds = [roberta_pred, xlnet_pred]

        a1 = open(args.xlnet_acc)
        xlnet_acc = json.load(a1)
        a2 = open(args.roberta_acc)
        roberta_acc = json.load(a2)

        accs = [xlnet_acc["acc"], roberta_acc["acc"]]

        fixed_alpha = 4
        auto_alpha = calc_autotune_alpha(accs)
        equal_alpha = 0

        print("Doing weighting on provided models")
        weighted_prob_fixed = aggregate_predictions(all_preds, fixed_alpha, accs)
        weighted_prob_auto = aggregate_predictions(all_preds, auto_alpha, accs)
        weighted_prob_equal = aggregate_predictions(all_preds, equal_alpha, accs)

        fixed_pred = post_processing(weighted_prob_fixed)
        auto_pred = post_processing(weighted_prob_auto)
        equal_pred = post_processing(weighted_prob_equal)

        json.dump(fixed_pred, open(output_path+"/unequal_weight_fixed_pred.json","w", encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(auto_pred, open(output_path+"/unequal_weight_auto_pred.json","w", encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(equal_pred, open(output_path+"/unequal_weight_equal_pred.json","w", encoding='utf-8'), ensure_ascii=False, indent=4)
        print('\nSuccessful json dump!')

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
    parser.add_argument('--xlnet_acc', help='path to saved xlnet accuracies after kfolds')
    parser.add_argument('--roberta_acc', help='path to saved roberta accuracies after kfolds')
    parser.add_argument('--output_path', help='path to save final prediction for test data')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
    print("Completed!")
