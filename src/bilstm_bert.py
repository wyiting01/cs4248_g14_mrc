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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials #for learning rate and batch size

torch.manual_seed(0)

# Default hyperparam
input_size=28
hidden_dim=256
num_layers=2
num_labels=10
batch_size=10
learning_rate=5e-5
num_epoch=10

def read_content(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read().split("\t")

class SquadDataset(Dataset):
    def __init__(self, in_path):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        contexts = read_content(f"{in_path}/context")
        questions = read_content(f"{in_path}/question")
        answers = read_content(f"{in_path}/answer")
        spans = read_content(f"{in_path}/answer_span")

        self.contexts = [ctx.strip() for ctx in contexts]
        self.questions = [qn.strip() for qn in questions]
        self.answers = [ans.strip() for ans in answers]
        self.spans = [span.strip().split() for span in spans]

        self.encodings = self.tokenizer(self.questions,
                                        self.contexts,
                                        truncation=True,
                                        padding="max_length",
                                        ## TODO: optimise hyperparam ltr
                                        max_length=384,
                                        stride=128,
                                        return_tensors="pt",
                                        add_special_tokens=True,
                                        return_overflowing_tokens=True,
                                        return_offsets_mapping=True
                                        )

        self.start_positions = []
        self.end_positions = []

        for i, (context, answer) in enumerate(zip(self.contexts, self.answers)):
            offsets = self.encodings['offset_mapping'][i]
            start_char_pos, end_char_pos = int(self.spans[i][0]), int(self.spans[i][1])

            # adjust for question-context pairs
            question_tokens = self.tokenizer.tokenize(self.questions[i])
            context_start_token_pos = len(question_tokens) + 2  # +1 for the [CLS] token

            if i == 24:
                print("Context:", self.contexts[i])
                print("Answer:", self.answers[i])
                print("Tokenized Answer:", self.tokenizer.tokenize(self.answers[i]))
                print("Span start:", start_char_pos, "Span end:", end_char_pos)
                print("Context tokens:", self.tokenizer.tokenize(self.contexts[i][start_char_pos:end_char_pos]))
            
            for j, (offset_start, offset_end) in enumerate(offsets[context_start_token_pos:]):
            
                token_start = token_end = None

                for j, (offset_start, offset_end) in enumerate(offsets[context_start_token_pos:]):  # Only look within the context offsets
                    # Check if the token starts or ends within the given answer span
                    
                    if offset_start == start_char_pos or (token_start is None and offset_start > start_char_pos):
                        token_start = j + context_start_token_pos
                    if offset_end == end_char_pos:
                        token_end = j + context_start_token_pos
                        break  # We can break early once we find the end token
                    
                    current_span = context[offset_start:offset_end]
                    if current_span == answer:
                        token_start = j + context_start_token_pos
                        token_end = token_start + len(self.tokenizer.tokenize(answer)) - 1  # Adjust for multi-token answers
                        break
                        
                if token_start is None or token_end is None:
                    raise ValueError(f"Token span for answer '{answer}' not found in context {i}.")
            
                self.start_positions.append(token_start)
                self.end_positions.append(token_end)

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "start_position": self.start_positions[idx],
            "end_position": self.end_positions[idx]
        }
        return item

class BERT_BiLSTM(nn.Module):
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
    criterion = nn.CrossEntropyLoss()  # Since we're predicting start and end positions

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
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
    
    train_set = SquadDataset(train_path)

    for i in range(5):
        sample = train_set[i]
        print("-----data_loaded-----")
        print("Context:", train_set.contexts[i])
        print("Tokenized Input IDs:", sample["input_ids"])
        print("Start Position:", sample["start_position"])
        print("End Position:", sample["end_position"])
        print("-----")

    test_set = SquadDataset(test_path)

    model = BERT_BiLSTM(input_size, hidden_dim, num_layers, num_labels).to(device)
    print(model)

    #TODO: change epoch val
    train(model, train_set, batch_size, learning_rate, num_epoch=1, device=device)
    
    acc = test(model, dataset=test_set, device=device)
    return {'loss': acc, 'status': STATUS_OK}

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

    # train_set = SquadDataset(train_path)
    # test_set = SquadDataset(test_path)

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