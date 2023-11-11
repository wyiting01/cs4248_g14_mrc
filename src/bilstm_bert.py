import argparse
import datetime
from itertools import islice
import numpy as np
import json
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

'''
Need to install libraries for hyperopt, numpy, sklearn, transformers:

pip install hyperopt
pip install numpy
pip install torch
pip install -U scikit-learn
pip install transformers

(From main folder cs4248_g14_mrc)
To train, use command:
python src/bilstm_bert.py --train --train_path data/curated/training_data/ --model_path bilstm.pt

To test, use command:
python src/bilstm_bert.py --test --test_path data/curated/test_data/ --model_path bilstm.pt
'''

# Default hyperparameters
hidden_dim=64
batch_size=16
learning_rate=5e-5
num_epoch=5
dropout_rate=0.1
seed = 0
num_splits = 2
n_best_size= 5

# Ensure no randomisation for every iteration of run.
def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_content(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read().split("\t")

class biLSTMDataset(Dataset):
    def __init__(self, in_path=None, x=None, y=None):
        print("Initialising dataset...")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

        if in_path != None:
            contexts = read_content(f"{in_path}/context")
            questions = read_content(f"{in_path}/question")
            answers = read_content(f"{in_path}/answer")
            spans = read_content(f"{in_path}/answer_span")
            question_ids = read_content(f"{in_path}/question_id")

            self.contexts = [ctx.strip() for ctx in contexts]
            self.questions = [qn.strip() for qn in questions]
            self.answers = [ans.strip() for ans in answers]
            self.spans = [span.strip().split() for span in spans]
            self.question_ids = [qn_id.strip() for qn_id in question_ids]
        else:
            self.contexts, self.questions, self.question_ids = zip(*x)
            self.answers, self.spans = zip(*y)
            
            # Initially cannot split because of how k fold works.
            self.spans = [span.split() for span in self.spans]

        self.encodings = self.tokenizer(self.questions,
                                        self.contexts,
                                        padding="max_length",
                                        max_length=384,
                                        stride=128,
                                        return_tensors="pt",
                                        truncation="only_second",
                                        add_special_tokens=True,
                                        return_overflowing_tokens=True,
                                        return_offsets_mapping=True
                                        )
        
        self.start_positions = []
        self.end_positions = []
        self.sample_mapping = self.encodings.pop("overflow_to_sample_mapping")
        self.offset_mapping = self.encodings.pop("offset_mapping")

        for i, offsets in enumerate(self.offset_mapping):
            input_ids = self.encodings["input_ids"][i].tolist()
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            sequence_ids = self.encodings["token_type_ids"][i]
            attention_mask = self.encodings["attention_mask"][i]

            sample_index = self.sample_mapping[i]
            context = self.contexts[sample_index]
            answer = self.answers[sample_index]
            start_char_pos, end_char_pos = map(int, self.spans[sample_index])

            # To adjust answer spans that might be off by one or two characters
            if context[start_char_pos:end_char_pos] != answer and context[start_char_pos-1:end_char_pos-1] == answer:
                start_char_pos -= 1
                end_char_pos -= 1
            elif context[start_char_pos-2:end_char_pos-2] == answer:
                start_char_pos -= 2
                end_char_pos -= 2

            # To find start and end of context
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            # sequence_ids will be None for [PAD]
            while token_end_index >= 0 and attention_mask[token_end_index] == 0:
                token_end_index -= 1
            # skip over the [SEP]
            if token_end_index >= 0 and input_ids[token_end_index] == self.tokenizer.sep_token_id:
                token_end_index -= 1

            # To find answers that are out of the span
            if not (offsets[token_start_index][0] <= start_char_pos and offsets[token_end_index][1] >= end_char_pos):
                self.start_positions.append(cls_index)
                self.end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char_pos:
                    token_start_index += 1
                self.start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char_pos:
                    token_end_index -= 1
                self.end_positions.append(token_end_index + 1)

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, i):
        idx_i = self.sample_mapping[i]

        item = {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "start_position": self.start_positions[i],
            "end_position": self.end_positions[i],
            "question_ids": self.question_ids[idx_i],
            "contexts": self.contexts[idx_i],
            "offset_mappings": self.offset_mapping[i]
        }
        return item

class BERT_BiLSTM(nn.Module):   
    def __init__(self, hidden_dim):
        super(BERT_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.bert_encoder = BertModel.from_pretrained('bert-base-cased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.start_out = nn.Linear(hidden_dim * 2, 1)
        self.end_out = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)

        lstm_out, _ = self.lstm(bert_outputs['last_hidden_state'])

        relu = self.relu(lstm_out)
        dropout_out = self.dropout(relu)

        start_logits = torch.abs(self.start_out(dropout_out)).squeeze(-1)
        end_logits = torch.abs(self.end_out(dropout_out)).squeeze(-1)
        
        return start_logits, end_logits

# Helper function to prepare data for k fold validations.
def extractAndMergeData(in_path):
    contexts = read_content(f"{in_path}/context")
    questions = read_content(f"{in_path}/question")
    answers = read_content(f"{in_path}/answer")
    spans = read_content(f"{in_path}/answer_span")
    question_ids = read_content(f"{in_path}/question_id")

    contexts = [ctx.strip() for ctx in contexts]
    questions = [qn.strip() for qn in questions]
    answers = [ans.strip() for ans in answers]
    # Cannot split because of how k fold needs flat arrays.
    spans = [span.strip() for span in spans]
    question_ids = [qn_id.strip() for qn_id in question_ids]

    return list(zip(contexts, questions, question_ids)), list(zip(answers, spans))

# train
def split_and_train(model, x_train, y_train, batch_size, learning_rate, num_epochs, device, model_path, test_set):
    # Perform Stratified K-Folds cross-validations.
    kfold = KFold(n_splits=num_splits)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Since we're predicting start and end positions

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    start = datetime.datetime.now()

    avg_losses_f = []
    avg_val_losses_f = []
    print("Beginning folding")

    for i, (train_index, valid_index) in enumerate(kfold.split(x_train, y_train)):
        seed_all(i * 100 + i)

        x_train_fold = np.array(x_train)[train_index]
        y_train_fold = np.array(y_train)[train_index]

        x_val_fold = np.array(x_train)[valid_index]
        y_val_fold = np.array(y_train)[valid_index]

        train_set = biLSTMDataset(x=x_train_fold, y=y_train_fold)
        valid_set = biLSTMDataset(x=x_val_fold, y=y_val_fold)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f'Fold {i + 1}')

        for epoch in range(num_epochs):
            model.train()
            avg_loss = 0.0
            for step, batch in enumerate(train_loader, 0):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
                start_positions = batch["start_positions"].to(device).long()
                end_positions = batch["end_positions"].to(device).long()

                optimizer.zero_grad()

                start_logits, end_logits = model(input_ids, attention_mask)

                #print(f"Batch {i} - Start Positions: {start_positions}")
                #print(f"Batch {i} - End Positions: {end_positions}")
                #print("Start log:", start_logits)
                #print("End log:", end_logits)

                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

                loss.backward()

                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

                if step % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, loss / 100))

            model.eval()

            avg_val_loss = 0

            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
                start_positions = batch["start_positions"].to(device).long()
                end_positions = batch["end_positions"].to(device).long()

                optimizer.zero_grad()

                start_logits, end_logits = model(input_ids, attention_mask)

                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

                avg_val_loss += loss.item() / len(valid_loader)

            if step % 100 == 99:
                print('Epoch[%d, %5d] average loss: %.3f average validation loss: %.3f' %
                    (epoch + 1, step + 1, loss / 100, avg_val_loss / 100))

        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)

        total_f1 = 0
        total_em = 0

        pred = {}
        scores = {}
        
        print("Conducting final test...")
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            question_ids = batch["question_ids"]
            contexts = batch["contexts"]
            offset_mappings = batch["offset_mappings"]
            
            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits = start_logits.cpu().detach().numpy() #grad included
            end_logits = end_logits.cpu().detach().numpy()
            #print("Shape of start_logits:", start_logits.shape)
            #print("Shape of end_logits:", end_logits.shape)

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

                print(f"Question ID: {qid}")
                print(f"Context: {ctxt}")
                print(f"Predicted Answer: {pred[qid]}")
                print(f"Top {n_best_size} predicted answers for Question ID {qid}:")
                for ans in scores[qid]:
                    print(f"Score: {ans['score']:.4f}, Text: {ans['text']}")
                
    end = datetime.datetime.now()

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))

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
    return pred, scores

def train(model, dataset, batch_size=batch_size, learning_rate=learning_rate, num_epoch=num_epoch, device='cpu', model_path=None):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Since we're predicting start and end positions

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 0):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            start_positions = batch["start_positions"].to(device).long()
            end_positions = batch["end_positions"].to(device).long()

            optimizer.zero_grad()

            start_logits, end_logits = model(input_ids, attention_mask)

            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            loss.backward()

            optimizer.step()

            total_loss += loss.item() / len(train_loader)
            
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, total_loss / 100))
    end = datetime.datetime.now()

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
    return {'loss': total_loss/len(train_loader), 'status': STATUS_OK}

# Pad to the right side
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    start_positions = [item["start_position"] for item in batch]
    end_positions = [item["end_position"] for item in batch]
    question_ids = [item["question_ids"] for item in batch]
    contexts = [item["contexts"] for item in batch]
    offset_mappings = [item["offset_mappings"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    
    padded_offset_mappings = pad_sequence(offset_mappings, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "start_positions": torch.tensor(start_positions).clone().detach(),
        "end_positions": torch.tensor(end_positions).clone().detach(),
        "offset_mappings": padded_offset_mappings,
        "question_ids": question_ids,
        "contexts": contexts
    }

# test
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

## test based on scores
def test_eval(model, dataset, n_best_size=n_best_size, device='cpu'):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
    pred = {}
    scores = {}

    print("Final testing on Test Dataset...")
    with torch.no_grad():
        for batch in test_loader:
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
            
            for i, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
            
                question_ids = batch["question_ids"]
                contexts = batch["contexts"]
                offset_mappings = batch["offset_mappings"]
            
                start_logits, end_logits = model(input_ids, attention_mask)

                start_logits = start_logits.cpu().detach().numpy() #grad included
                end_logits = end_logits.cpu().detach().numpy()
                #print("Shape of start_logits:", start_logits.shape)
                #print("Shape of end_logits:", end_logits.shape)

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

                    # Print out the predicted answer for a given question and context.
                    print(f"Question ID: {qid}")
                    print(f"Context: {ctxt}")
                    print(f"Predicted Answer: {pred[qid]}")
                    print(f"Top {n_best_size} predicted answers for Question ID {qid}:")
                    for ans in scores[qid]:
                        print(f"Score: {ans['score']:.4f}, Text: {ans['text']}")
                
    print("Final testing completed. Best answers computed.")
    return pred, scores

def main(args):
    train_path = args.train_path
    test_path = args.test_path
    model_path = args.model_path
    output_path = args.output_path
    score_path = args.score_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BERT_BiLSTM(hidden_dim).to(device)

    if args.train:
        #test_outputs, test_scores = split_and_train(model, x_train, y_train, batch_size, learning_rate, num_epoch, device, model_path, test_set)
        train_set = biLSTMDataset(train_path)

        train(model, train_set, batch_size=batch_size, learning_rate=learning_rate, num_epoch=num_epoch, device=device, model_path=model_path)

    if args.test:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_set = biLSTMDataset(test_path)
    
        test_outputs, test_scores = test_eval(model, dataset=test_set, n_best_size = n_best_size, device=device)

        json.dump(test_outputs, open(output_path,"w"), ensure_ascii=False, indent=4)
        json.dump(test_scores, open(score_path,"w"), ensure_ascii=False, indent=4)
        print('\nSuccessful json dump!')

    print('\n==== All done ====')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--train_path', help='path to the training datasets')
    parser.add_argument('--test_path', help='path to the test datasets')
    parser.add_argument('--model_path', help='path to where the model is saved')
    parser.add_argument('--output_path', default="pred.json", help='path to model_prediction')
    parser.add_argument('--score_path', default="scores.json", help='path to model scores')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
