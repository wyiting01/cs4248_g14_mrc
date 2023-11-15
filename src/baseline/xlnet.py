import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import XLNetForQuestionAnswering, XLNetTokenizerFast
from sklearn.model_selection import KFold

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
    
def train(model, dataset, batch_size=16, learning_rate=5e-5, num_epoch=10, device='cpu', model_path=None):

    print("Training XLNet Model")

    data_loader = DataLoader(dataset, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()

        for step, batch in enumerate(data_loader, 0):
            
            # get the inputs; batch is a dict
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            # calculate the loss
            loss = outputs[0]

            # do backward propagation
            loss.backward()

            # do the parameter optimization
            optimizer.step()

            # print loss value every 100 iterations and reset running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, loss / 100))

    end = datetime.datetime.now()

    # save trained model
    checkpoint = {
        'epoch': num_epoch,
        'lr': learning_rate,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    #return checkpoint

def calc_f1(p, t):
    p_tokens = p.split() # predicted
    t_tokens = t.split() # true
    common_tokens = set(p_tokens) & set(t_tokens)
    if not p_tokens or not t_tokens:
        return 0
    precision = len(common_tokens) / len(p_tokens)
    recall = len(common_tokens) / len(t_tokens)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def test(model, dataset, n_best_size=20, max_answer_length=30, device='cpu'):
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    pred_all = {}
    pred_top = {}

    metrics = {}
    correct_pred = 0
    f1_scores = []

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
                curr_answer = answer[i]

                valid_answers = {}
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
                            pred_answer = ctxt[start_char: end_char] 
                            pred_score = start_logits[start] + end_logits[end]
                            if valid_answers.get(pred_answer) == None or float(valid_answers.get(pred_answer)) < pred_score:
                                valid_answers[pred_answer] = str(pred_score)

                valid_answers = dict(sorted(valid_answers.items(), key=lambda x: float(x[1]), reverse=True)[:n_best_size])
                if len(valid_answers) == 0:
                    pred_all[qid] = {}
                    pred_top[qid] = ""
                else:
                    pred_all[qid] = valid_answers
                    pred_top[qid] = next(iter(valid_answers))

                # Calculate accuracy
                if pred_top[qid] == curr_answer:
                    correct_pred += 1

                    # Calculate F1 score
                f1 = calc_f1(pred_top[qid], curr_answer)
                f1_scores.append(f1)


    metrics['acc'] = correct_pred / len(f1_scores)
    metrics['f1'] = sum(f1_scores) / len(f1_scores)

    return pred_all, pred_top, metrics

def cross_val_worker(fold, train_index, test_index, dataset, model, device, model_path=None, batch_size=16, collate_fn=None): 
    print(f"Processing fold {fold + 1}...")

    # # Create subsets 
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)

    train(model, train_subset, device=device, model_path=model_path)
    pred_all, pred_top, metrics = test(model, dataset=test_subset, device=device)
    return metrics

def main(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Initialising XLNet Model")
    model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(device)

    if args.train:
        train_path, model_path = args.data_path, args.model_path
        squad_train = SquadDataset(train_path)

        # specify hyperparameters
        num_epoch = 2
        batch_size = 16
        learning_rate = 5e-5

        # train the model
        train(model=model, dataset=squad_train, num_epoch=num_epoch, batch_size=batch_size, learning_rate=learning_rate, device=device, model_path=model_path)
    
    if args.test:
        test_path, model_path, output_path = args.data_path, args.model_path, args.output_path
        print("Loading Saved Weights from Training")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        squad_test = SquadDataset(test_path)

        # specify hyperparameters
        n_best_size = 20
        max_answer_length = 30

        # use trained model to make predictions
        pred_all, pred_top, metrics = test(model=model, dataset=squad_test, n_best_size=n_best_size, max_answer_length=max_answer_length, device=device)
    
        # write model prediction into json file
        with open(output_path + "/xlnet_pred_all.json", 'w') as f:
            json.dump(pred_all, f)

        with open(output_path + "/xlnet_pred_top.json", 'w') as f:
            json.dump(pred_top, f)

        print('Model predictions saved in ', output_path)

    if args.train_kf:
        k=5

        train_path, metric_path, model_path = args.data_path, args.metric_path, args.model_path

        squad_train = SquadDataset(train_path)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        metric_sums = {'acc': 0, 'f1': 0}

        for fold, (train_index, test_index) in enumerate(kf.split(squad_train)):
            #cross_val_worker(fold, train_index, test_index, dataset, model, device, model_path=None, batch_size=16, collate_fn=None)
            kf_model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(device)
            fold_metrics = cross_val_worker(fold, train_index=train_index, test_index=test_index, dataset=squad_train, model=kf_model, device=device, model_path=model_path)
            print(fold_metrics)

            for key in metric_sums:
                metric_sums[key] += fold_metrics[key]

        cval_metrics = {metric: total / k for metric, total in metric_sums.items()}

        json.dump(cval_metrics, open(metric_path,"w"), ensure_ascii=False, indent=4)
        

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--train_kf', default=False, action='store_true', help='perform kfold on model')
    parser.add_argument('--data_path', help='path to the dataset file')
    parser.add_argument('--model_path', help='path to save trained model')
    parser.add_argument('--output_path', default="pred.json", help='path to model_prediction')
    parser.add_argument('--metric_path', default="xlnet_metrics.json", help='path to model metrics from cv')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
    print("Completed!")
