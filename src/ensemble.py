import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from roberta import RobertaTokenizerFast, RobertaForQuestionAnswering, SquadDataset as robertSquadDataset
from xlnet import XLNetForQuestionAnswering, XLNetTokenizerFast, SquadDataset as xlnetSquadDataset

"""
multiplicative: python ensemble --multiplicative --data_path "../data/curated/test_data" --roberta_path "./roberta.pt" --xlnet_path "./xlnet.pt" --output_path "./mul_pred_roberta_xlnet.json"
maximum: python ensemble --maximum --data_path "../data/curated/test_data" --roberta_path "./roberta.pt" --xlnet_path "./xlnet.pt" --output_path "./max_pred_roberta_xlnet.json"
"""

def test_reoberta(model, dataset, n_best_size=20, device='cpu'):
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

def get_answer(model1_pred, model2_pred, kind, max_answer_length):
    """
    input:
    1. model1's prediction
    2. model2's prediction
    3. kind of equal weightage: "multiplicative" or "maximum"

    output:
    1. final prediction - {qid: answer}
    """
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
            if kind == "multiplicative": score = model1_pred[question]["start"][start] * model2_pred[question]["start"][start]
            elif kind == "maximum": score = max(model1_pred[question]["start"][start], model2_pred[question]["start"][start])
            common_start[start] = score
        for end in common_end_keys:
            if kind == "multiplicative": score = model1_pred[question]["end"][end] * model2_pred[question]["end"][end]
            elif kind == "maximum": score = max(model1_pred[question]["end"][end], model2_pred[question]["end"][end])
            common_end[end] = score

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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplicative', default=False, action='store_true', help='ensemble via multiplcative')
    parser.add_argument('--maximum', default=False, action='store_true', help='ensemble via maximum')
    parser.add_argument('--data_path', help='path to the dataset file')
    parser.add_argument('--roberta_path', help='path to load trained roberta')
    parser.add_argument('--xlnet_path', help='path to load trained xlnet')
    parser.add_argument('--output_path', default="ensemble_pred.json", help='path to model_prediction')
    return parser.parse_args()

def main(args):
    data_path, roberta_path, xlnet_path, output_path = args.data_path, args.roberta_path, args.xlnet_path, args.output_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    n_best_size = 20
    max_answer_length = 50

    # roberta
    print("Started on RoBERTa")
    roberta_data = robertSquadDataset(data_path)
    roberta_dict = torch.load(roberta_path, map_location=torch.device(device))
    roberta = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(device)
    roberta.load_state_dict(roberta_dict["model_state_dict"])
    roberta_pred = test_reoberta(roberta, roberta_data, n_best_size, device=device)

    # xlnet
    print("Started on XLNet")
    xlnet_data = xlnetSquadDataset(data_path)
    xlnet_dict = torch.load(xlnet_path, map_location=torch.device(device))
    xlnet = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(device)
    xlnet.load_state_dict(xlnet_dict["model_state_dict"])
    xlnet_pred = test_xlnet(xlnet, xlnet_data, n_best_size, device=device)

    if args.multiplicative:
        final_prediction = get_answer(roberta_pred, xlnet_pred, kind="multiplicative", max_answer_length=max_answer_length)
    if args.maximum:
        final_prediction = get_answer(roberta_pred, xlnet_pred, kind="maximum", max_answer_length=max_answer_length)

    # write model prediction into json file
    with open(output_path, 'w') as f:
        json.dump(final_prediction, f)
    print('Model predictions saved in ', output_path)
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)
    print("Completed!")