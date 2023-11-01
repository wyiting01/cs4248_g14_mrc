import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from sklearn.model_selection import RepeatedKFold

# Default hyperparameters
input_size=28
hidden_dim=256
num_layers=2
num_labels=10
batch_size=10
learning_rate=5e-5
num_epoch=10
dropout_rate=0.1

# Ensure no randomisation for every iteration of run.
def seed_all(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_content(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read().split("\t")

class biLSTMDataset(Dataset):
    def __init__(self, in_path):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        contexts = read_content(f"{in_path}/context")
        questions = read_content(f"{in_path}/question")
        answers = read_content(f"{in_path}/answer")
        spans = read_content(f"{in_path}/answer_span")
        question_ids = read_content(f"{in_path}/question_id")

        self.contexts = [ctx.strip() for ctx in contexts]
        self.questions = [qn.strip() for qn in questions]
        self.answers = [ans.strip() for ans in answers]
        self.spans = [span.strip().split() for span in spans]
        self.question_ids = [id.strip() for id in question_ids]

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
            sequence_ids = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
            
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
            while sequence_ids[token_end_index] != 1:
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
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "start_position": self.start_positions[idx],
            "end_position": self.end_positions[idx],
            "question_ids": self.question_ids[idx]
        }
        return item

class BERT_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_labels):
        super(BERT_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.start_out = nn.Linear(hidden_dim * 2, 1)
        self.end_out = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_outputs['last_hidden_state'])
        max_pooled = F.adaptive_max_pool1d(lstm_out.permute(0, 2, 1), 1).squeeze(-1)
        relu_out = self.relu(max_pooled)
        dropout_out = self.dropout(relu_out)

        start_logits = self.start_out(dropout_out).squeeze(-1)
        end_logits = self.end_out(dropout_out).squeeze(-1)
        return start_logits, end_logits

# train
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

def train(model, dataset, batch_size=batch_size, learning_rate=learning_rate, num_epoch=num_epoch, device='cpu', model_path=None):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            start_positions = batch["start_positions"].float().to(device)
            end_positions = batch["end_positions"].float().to(device)
            
            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Compute loss and backpropagate
            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epoch}, Loss: {total_loss/len(data_loader)}")
    end = datetime.datetime.now()

    # checkpoint = {
    #     'epoch': num_epoch,
    #     'lr': learning_rate,
    #     'batch_size': batch_size,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }
    # torch.save(checkpoint, model_path)

    # print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    return {'loss': total_loss/len(data_loader), 'status': STATUS_OK}

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

def test(model, dataset, device='cpu'):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
    
    total_f1 = 0
    total_em = 0

    # pred = {}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start = batch["start_position"].to(device)
            end = batch["end_position"].to(device)
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

# Training for best hyperparam config
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
    'batch_size': hp.choice('batch_size', [1, 8, 16, 32, 64]),
    # 'num_layers': hp.quniform('num_layers', 1, 5, 1),
    # 'hidden_dim': hp.quniform('hidden_dim', 64, 512, 64),
    # 'max_length': hp.choice('max_length', list(range(256, 513, 64))),
    # 'stride': hp.choice('stride', list(range(64, 257, 64)))
}

def objective(params):
    # num_layers = int(params['num_layers'])
    # hidden_dim = int(params['hidden_dim'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])

    train_path = args.train_path
    test_path = args.test_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_set = biLSTMDataset(train_path)

    test_set = biLSTMDataset(test_path)

    model = BERT_BiLSTM(input_size, hidden_dim, num_layers, num_labels).to(device)
    print(model)

    #TODO: change epoch val
    train(model, train_set, batch_size, learning_rate, num_epoch=1, device=device)
    
    acc = test(model, dataset=test_set, device=device)
    return {'loss': acc, 'status': STATUS_OK}

## SPLIT FOR KFOLDS HERE
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

## FINAL TEST AGAIN FOR EVAL
def test_eval(model, dataset, device='cpu'):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
    
    total_f1 = 0
    total_em = 0

    pred = {}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start = batch["start_position"].to(device)
            end = batch["end_position"].to(device)
            question_ids = batch["question_ids"].to(device)

            model_output = model(input_ids, attention_mask)

            for i in range(input_ids.size(0)):
                start_logits = model_output.start_top_log_probs[i].cpu().detach().numpy()
                end_logits = model_output.end_top_log_probs[i].cpu().detach().numpy()

                # Getting the most likely start and end positions
                start_pos = torch.argmax(start_logits, dim=1)
                end_pos = torch.argmax(end_logits, dim=1)

                pred_answer = dataset.tokenizer.decode(input_ids[i, start_pos[i]:end_pos[i]+1])
                true_answer = dataset.answers[i]
                print("-----TEST-----")
                print(f"Sample {i+1}:")
                print("Predicted Answer:", pred_answer)
                print("True Answer:", true_answer)
                if pred_answer == true_answer:
                    total_em += 1
                total_f1 += get_f1(pred_answer, true_answer)

                question_id = question_ids[i]

    print(total_em)
    n = len(dataset)
    avg_f1 = total_f1/n
    avg_em = total_em/n

    # return total_start_pos, total_end_pos
    # TODO: check which metric to use, and if weighted average is a 
    # return avg_f1, avg_em 
    return -(0.5 * avg_f1 + 0.5 * avg_em)

def main(args):
    train_path = args.train_path
    test_path = args.test_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # TODO: update max_evals ltr
    optimised_hyperparam = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1)
    print("After hyperopt: " + str(optimised_hyperparam))

    ### train/test w final optimised hyperparams

    # train(model, train_set, num_epoch=10, batch_size=16, device=device) 
    # test_outputs = test(model, dataset=test_set, device=device)
    # print(test_outputs)

    # train_set = biLSTMDataset(train_path)
    # test_set = biLSTMDataset(test_path)

    # model = BERT_BiLSTM(input_size, hidden_dim, num_layers, num_labels).to(device)
    # train(model, train_set, num_epoch=10, batch_size=16, device=device)
    
    # test_outputs = test(model, dataset=test_set, device=device)
    # print(test_outputs)

    print('\n==== All done ====')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True, help='path to the training datasets')
    parser.add_argument('--test_path', required=True, help='path to the test datasets')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)