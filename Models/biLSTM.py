#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import contractions
import copy
import datetime
import numpy as np
import os
import random
import requests
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import RepeatedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel

'''
Need to install libraries for numpy, contractions, sklearn, transformers:

pip install numpy
pip install contractions
pip install -U scikit-learn
pip install transformers

To run, use command: python3 Models/biLSTM.py --train_path data/curated/training_data/ --test_path data/curated/test_data/ --model_path model.pt
'''

# Pytorch version: Adapted from https://www.kaggle.com/code/mlwhiz/bilstm-pytorch-and-keras

# Randomly initialised hyperparameters. To be put into a grid search.
embed_size = 64
maxQnLen = 80
batch_size = 64
num_epochs = 1
num_splits = 2
hidden_size = 64
seed = 0
dropout = 0.1
learning_rate = 0.001

#puncts = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '[', ']', '{', '}', '\\', "\|", ':', ';', '\'', '\"', '<', ',', '>', '.', '?', '/']

# Adapted from: https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english
url ="https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
american_to_british_dict = requests.get(url).json()

# Ensure no randomisation for every iteration of run.
def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class biLSTMDataset(Dataset):
    def __init__(self, x=None, y=None, input_path="", isTraining=False):
        if input_path != "":
            self.isDatasetAvail = False

            # Initialise training set.
            context = input_path + "/context"
            questions = input_path + "/question"
            answers = input_path + "/answer"
            answer_spans = input_path + "/answer_span"

            print('Initialising dataset')

            self.contexts = self.extractAndCleanData(context)
            self.questions = self.extractAndCleanData(questions)
            self.answers = self.extractAndCleanData(answers)
            self.answer_spans = self.extractAndCleanData(answer_spans)

            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

            print('Tokenising texts')

            self.encodings = self.tokenizer(self.contexts, self.questions, truncation=True, padding=True, return_tensors="pt")

            print('Tokenising complete')

            print('Tokenising targets')

            self.start_positions = []
            self.end_positions = []

            for i, (context, answer) in enumerate(zip(self.contexts, self.answers)):
                answer_start_idx = context.find(answer)
                answer_end_idx = answer_start_idx + len(answer) - 1
                start_position = self.char_to_token_position(i, answer_start_idx)
                end_position = self.char_to_token_position(i, answer_end_idx)
                self.start_positions.append(start_position)
                self.end_positions.append(end_position)
        else:
            self.isDatasetAvail = True
            self.x = x
            self.y = y

            self.longest_len = 0
            for i in self.x:
                if len(i) > self.longest_len:
                    self.longest_len = len(i)

    def char_to_token_position(self, idx, char_position):
        # return len(self.tokenizer.encode(self.contexts[idx][:char_position]))
        tokens_info = self.tokenizer.encode_plus(self.contexts[idx], add_special_tokens=True, return_offsets_mapping=True)
        offsets = tokens_info['offset_mapping']
    
        for token_pos, (start, end) in enumerate(offsets):
            if char_position >= start and char_position < end:
                return token_pos
        print("Error in span")

        # Cannot return None if using k-fold validations.
        return len(offsets)

    def extractAndCleanData(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            data = file.read().split('\t')
        print('Cleaning dataset')
        newDataset = [self.clean_text(line.strip()) for line in data]
        return newDataset

    # Clean up input text to be more standardised (spelling & word form) and decrease dictionary size.
    # Note: Due to nature of questions & answers, cannot remove all special characters as they are present in them.
    def clean_text(self, text):
        text = text.lower()

        # Clean non-ASCII dashes.
        text = self.clean_dashes(text)
        # Clean non-ASCII apostrophes and quotation marks.
        text = self.clean_contractions(text)
        # This expands any contractions: (e.g. I'd --> I had)
        text = contractions.fix(text)

        text = self.correct_spelling(text, american_to_british_dict)

        return text

    # Remove non-ASCII dashes.
    def clean_dashes(self, text):
        text = text.replace(u'\u2013', '-')
        return text

    # Remove non-ASCII quotes.
    def clean_contractions(self, text):
        specials = [u'\u2018', u'\u2019', u'\u00B4', u'\u0060']
        for s in specials:
            text = text.replace(s, "'")
        return text

    def correct_spelling(self, text, dic):
        words = text.split()
        for i in range(len(words)):
            if words[i] not in dic.keys():
                continue
            else:
                words[i] = american_to_british_dict[words[i]]
        text = ' '.join(words)
        return text

    def get_texts_and_questions(self):
        return self.encodings["input_ids"]

    def get_labels_vocab(self):
        return list(zip(self.start_positions, self.end_positions))

    def __getitem__(self, idx):
        if not self.isDatasetAvail:
            item = {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "start_position": self.start_positions[idx],
                "end_position": self.end_positions[idx]
            }
        else:
            item = {
                "input_ids": self.x[idx],
                "attention_mask": torch.tensor([1] * len(self.x[idx]) + [0] * (self.longest_len - len(self.x[idx]))),
                "start_position": self.y[idx][0][0],
                "end_position": self.y[idx][0][1]
            }
        return item

    def __len__(self):
        if not self.isDatasetAvail:
            return len(self.questions)
        else:
            return len(self.x)

class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_labels):
        super(BERT_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        # self.classifier = nn.Linear(hidden_dim*2, num_labels)  # multiply for bidirectional
        self.start_classifier = nn.Linear(hidden_dim*2, 1) 
        self.end_classifier = nn.Linear(hidden_dim*2, 1)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.bilstm(bert_outputs['last_hidden_state'])
        # logits = self.classifier(lstm_out)
        start_logits = self.start_classifier(lstm_out).squeeze(-1)
        end_logits = self.end_classifier(lstm_out).squeeze(-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)  # Returns two tensors of shape [batch_size, seq_length]

# Pad to the right side
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    start_positions = [item["start_position"] for item in batch]
    end_positions = [item["end_position"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "start_positions": torch.tensor(start_positions),
        "end_positions": torch.tensor(end_positions)
    }

def split_and_train(model, dataset, batch_size, learning_rate, num_epochs, device, model_path):
    print('Starting training')
    seed_all()

    x_train = dataset.get_texts_and_questions()
    y_train = torch.tensor(dataset.get_labels_vocab())

    print(len(x_train), len(y_train))
    assert(len(x_train) == len(y_train))

    #dataloader = DataLoader(dataset, batch_size=batch_size)

    # Combines a Sigmoid layer and the BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Perform Stratified K-Folds cross-validations.
    kfold = RepeatedKFold(n_splits=num_splits, random_state=seed)

    for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
        seed_all(i * 100 + i)

        x_train_fold = x_train[train_index.astype(int)].to(torch.long)
        y_train_fold = y_train[train_index.astype(int), np.newaxis].to(torch.float32)

        x_val_fold = x_train[valid_index.astype(int)].to(torch.long)
        y_val_fold = y_train[valid_index.astype(int), np.newaxis].to(torch.float32)

        train_set = biLSTMDataset(x=x_train_fold, y=y_train_fold)
        valid_set = biLSTMDataset(x=x_val_fold, y=y_val_fold)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        print(f'Fold {i + 1}')

        start_time = datetime.datetime.now()

        for epoch in range(num_epochs):
            model.train()
            avg_loss = 0.0
            for step, batch in enumerate(train_loader, 0):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                optimizer.zero_grad()

                start_logits, end_logits = model(input_ids, attention_mask)

                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

                loss.backward()

                optimizer.step()

                avg_loss = loss.item() / len(dataloader)

                if step % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, loss / 100))

            model.eval()

            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((dataset.__len__()))

            avg_val_loss = 0

            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                optimizer.zero_grad()

                start_logits, end_logits = model(input_ids, attention_mask)

                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

                avg_loss = loss.item() / len(valid_loader)
                valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]

            if step % 100 == 99:
                print('Epoch[%d, %5d] average loss: %.3f average validation loss: %.3f' %
                    (epoch + 1, step + 1, loss / 100, avg_val_loss / 100))

        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)

        test_outputs = test(model, dataset=test_dataset, device=device)
        print(test_outputs)

    end_time = datetime.datetime.now()

    total_time = (end_time - start_time).seconds / 60.0

    print('Training finished in {} minutes.'.format(total_time))

    checkpoint = {
        'epoch': num_epochs,
        'lr': learning_rate,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, model_path)

    print("Model saved in ", model_path)

def test(model, dataset, device='cpu'):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
    
    # total_start_pos = []
    # total_end_pos = []

    total_f1 = 0
    total_em = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            start_logits, end_logits = model(input_ids, attention_mask)

            # Getting the most likely start and end positions
            start_pos = torch.argmax(start_logits, dim=1)
            end_pos = torch.argmax(end_logits, dim=1)
            
            # total_start_pos.extend(start_pos.cpu().numpy())
            # total_end_pos.extend(end_pos.cpu().numpy())

            for i in range(input_ids.size(0)):
                pred_answer = dataset.tokenizer.decode(input_ids[i, start_pos[i]:end_pos[i]+1])
                true_answer = dataset.answers[i]
                print("-----TEST-----")
                print(f"Sample {i+1}:")
                print("Predicted Answer:", pred_answer)
                print("True Answer:", true_answer)
                print("-----")
                if pred_answer == true_answer:
                    total_em += 1
                total_f1 += get_f1(pred_answer, true_answer)
    print(total_em)
    n = len(dataset)
    avg_f1 = total_f1/n
    avg_em = total_em/n

    # return total_start_pos, total_end_pos
    # TODO: check which metric to use, and if weighted average is a 
    # return avg_f1, avg_em 
    return -(0.5 * avg_f1 + 0.5 * avg_em)

def get_f1(p, t):
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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True, help='path to the training files')
    parser.add_argument('--test_path', help='path to the test files')
    parser.add_argument('--model_path', help='path to save trained model')
    parser.add_argument('--output_path', help='path to model_prediction')
    return parser.parse_args()


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    args = get_arguments()

    input_path = args.train_path

    model_path = args.model_path

    test_path = args.test_path

    seed_all()

    start_time = time.time()

    dataset = biLSTMDataset(input_path=input_path)

    test_dataset = biLSTMDataset(input_path=test_path)

    print("Time for initialisation: ", time.time() - start_time)

    model = biLSTM().to(device)

    split_and_train(model, dataset, batch_size, learning_rate, num_epochs, device, model_path=model_path)

    # test the model (to be modified from trg data to test data)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    squad_test = SquadDataset(test_path)
    test_output = test(model, dataset=squad_test, device=device)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
