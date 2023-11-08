'''
python ensemble_optuna.py --train --xlnet --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../xlnet.json" --albert_dict "../albert.json"
python ensemble_optuna.py --train --albert --data_path "../data/curated/ensemble_data/train" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../xlnet.json" --albert_dict "../albert.json"

python ensemble_weighting.py --get_candidates --xlnet --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../ensemble/xlnet_val.json" --albert_dict "../ensemble/albert_val.json.json"
python ensemble_weighting.py --get_candidates --albert --data_path "../data/curated/ensemble_data/val" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../ensemble/xlnet_val.json" --albert_dict "../ensemble/albert_val.json"

python ensemble_weighting.py --get_candidates --xlnet --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../ensemble/xlnet_test.json" --albert_dict "../ensemble/albert_test.json"
python ensemble_weighting.py --get_candidates --albert --data_path "../data/curated/test_data" --xlnet_path "../model/xlnet.pt" --albert_path "../model/albert.pt" --xlnet_dict "../ensemble/xlnet_test.json" --albert_dict "../ensemble/albert_test.json"
python ensemble_weighting.py --test --xlnet_dict "../ensemble/xlnet.json" --albert_dict "../ensemble/albert.json" --output_path "../ensemble/final.json" --xlnet_weight 0.76 --albert_weight 0.24
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
from transformers import AlbertForQuestionAnswering, AlbertTokenizerFast, XLNetForQuestionAnswering, XLNetTokenizerFast
import xlnet
import albert
import ensemble_optuna

def test_albert(model, dataset, n_best_size=20, device='cpu'):
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

def post_processing(score_dict, max_answer_length=150):
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

        if args.albert:
            print("Initialising Albert Model")
            albert_model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2").to(device)
            albert_dataset = albert.SquadDataset(args.data_path)

            # train the albert model
            albert.train(model=albert_model, dataset=albert_dataset, num_epoch=num_epoch, batch_size=batch_size, learning_rate=learning_rate, device=device, model_path=args.albert_path)
    
    elif args.get_candidates:
      
        if args.albert:
        
            albert_model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2").to(device)
            albert_checkpt = torch.load(args.albert_path)
            albert_model.load_state_dict(albert_checkpt['model_state_dict'])
            albert_validation = albert.SquadDataset(args.data_path)

            albert_pred = test_albert(albert_model, albert_validation, n_best_size=20, device=device)

            with open(args.albert_dict, 'w') as f:
                json.dump(albert_pred, f)

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
        f2 = open(args.albert_dict)
        albert_pred = json.load(f2)

        print("Doing weighting on xlnet and albert")
        weighted_score_dict = ensemble_optuna.weighting_score(xlnet_pred, albert_pred, float(args.xlnet_weight), float(args.albert_weight))
        final_pred = post_processing(weighted_score_dict)

        with open(args.output_path, 'w') as f:
            json.dump(final_pred, f)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--get_candidates', default=False, action='store_true', help='get candidates for test data')
    parser.add_argument('--test', default=False, action='store_true', help='generating output for test data')
    parser.add_argument('--xlnet', default=False, action='store_true', help='get xlnet dict')
    parser.add_argument('--albert', default=False, action='store_true', help='get albert dict')
    parser.add_argument('--data_path', help='path to the dataset file')
    parser.add_argument('--xlnet_path', help='path to save trained xlnet model')
    parser.add_argument('--albert_path', help='path to save trained albert model')
    parser.add_argument('--xlnet_dict', help='path to save xlnet pred on val')
    parser.add_argument('--albert_dict', help='path to save albert pred on val')
    parser.add_argument('--output_path', help='path to save final output')
    parser.add_argument('--albert_weight', default=0.5, help='path to save albert pred on val')
    parser.add_argument('--xlnet_weight', default=0.5, help='path to save final output')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
    print("Completed!")
    