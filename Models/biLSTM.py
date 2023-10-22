import datetime
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

from gensim.models import Word2Vec
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

# Pytorch version: Adapted from https://www.kaggle.com/code/mlwhiz/bilstm-pytorch-and-keras

# Randomly initialised hyperparameters. To be put into a grid search.
embed_size = 360
max_features = 100000
maxQnLen = 80
batch_size = 64
num_epochs = 10
hidden_size = 128
seed = 0
dropout = 0.1

# Ensure no randomisation for every iteration of run.
def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class biLSTMDataset(Dataset):
    def __init__(self, input_path, isTraining=False):
        # Initialise training set.
        context = input_path + "/context"
        questions = input_path + "/question"
        answers = input_path + "/answer"
        answer_spans = input_path + "/answer_span"

        self.context = extractData(context)
        self.questions = extractData(questions)
        self.answers = extractData(answers)
        self.answer_spans = extractData(answer_spans)

        X_train = self.context + self.questions

        # Tokenise sentences.
        tokeniser = Tokenizer(num_words=max_features)
        # Since input is text.
        tokeniser.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)

        # Pad the sentences.
        X_train = pad_sequences(X_train, maxlen=maxQnLen)

        self.text_vocab = X_train

        target_labels = [[0]* len(seq) for seq in X_train]
        for i in range(len(target_labels)):
            start_index, end_index = self.answer_spans[i][0], self.answer_spans[i][1]
            target_labels[i][start_index:end_index + 1] = [1] * (end_index - start_index + 1)
        
        self.labels_vocab = target_labels
        
    def extractData(file_path):
        with open(file_path, "r") as file:
            data = file.read().split("\t")
        newDataset = [line.strip() for line in data]
        return newDataset

    def __getitem__(self, index):
        data, target = self.text_vocab[index], self.labels_vocab[index]
        return data, target

    def __len__(self):
        return len(self.questions)

    '''
    Unused but possibly useful pre-processing functions to increase accuracy.
    def known_contractions(embed):
        known = []
        for contract in contraction_mapping:
            if contract in embed:
                known.append(contract)
        return known

    def clean_contractions(text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    def correct_spelling(x, dic):
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x

    def unknown_punct(embed, punct):
        unknown = ''
        for p in punct:
            if p not in embed:
                unknown += p
                unknown += ' '
        return unknown

    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])
    
        for p in punct:
            text = text.replace(p, f' {p} ')
    
        return text

    def add_lower(embedding, vocab):
        count = 0
        for word in vocab:
            if word in embedding and word.lower() not in embedding:
                embedding[word.lower()] = embedding[word]
                count += 1
        print(f"Added {count} words to embedding")

    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '°', '€', '›', '?', '?', '?', 'Â', '½', 'à', '…', 
     '“', '”', '–', 'â', '?', '¢', '²', '¬', '?', '±', '¿', '?', '?', '¦', '?', '?', '¥', '—', '‹', '?', '?', '¼', '’', '¨', 'é', '¯', 'è', '¸', 'Ã', '?', '‘', '?', 
     '?', '?', '?']

    def clean_text(x):
        x = str(x)
        for punct in puncts:
            if punct in x:
                x = x.replace(punct, f' {punct} ')
        return x


    def clean_numbers(x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
        return x

    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

    def _get_contraction(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re

    contractions, contraction_re = _get_contraction(contraction_dict)
    def replace_typical_contraction(text):
        def replace(match):
            return contractions[match.group(0)]
        return contract_re.sub(replace, text)
    '''

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        drp = dropout
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4 , self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        #print("avg_pool", avg_pool.size())
        #print("max_pool", max_pool.size())
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

def split_and_train(model, dataset, batch_size, learning_rate, num_epochs, device='cpu'):
    seed_all()

    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Combines a Sigmoid layer and the BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    optimiser = optim.Adam(model.parameters, lr=learning_rate)
    
    '''
    # Perform Stratified K-Folds cross-validations.
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train, y_train))

    for i, (train_index, valid_index) in enumerate(splits):
        seed_all(i * 100 + i)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(x_train[train_index.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_index.astype(int), np.newaxis], dtype=torch.float32).cuda()

        if feats:
            kfold_X_features = features[train_index.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        
        model = copy.deepcopy(model_obj)

        model.cuda()

        train_set = Dataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))
        valid_set = Dataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        print(f'Fold {i + 1}')
    '''
    start_time = datetime.datetime.now()

    for epoch in range(num_epoch):
        model.train()
        avg_loss = 0.0
        for step, data in enumerate(dataloader, 0):
            texts, labels = data[0].to(device), data[1].to(device)

            optimiser.zero_grad()

            outputs = model(texts)

            loss = criterion(outputs)

            loss.backward()

            optimiser.step()

            avg_loss = loss.item() / len(dataloader)

            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, loss / 100))

        model.eval()

    end_time = datetime.datetime.now()

    total_time = (end_time - start_time).seconds / 60.0

    print('Training finished in {} minutes.'.format(total_time))

    checkpoint = {
        'model_state_dict': best_trained_model.state_dict(),
        'vocab': dataset.getVocab(),
        'best_params': best_params,
        'model': best_trained_model
    }

def test(model, dataset, device, batch_size):
    seed_all()

    test_dataLoader = DataLoader(dataset, batch_size=batch_size)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='path to the text file')
    return parser.parse_args()

def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    args = get_arguments()

    input_path = args.input_path

    seed_all()

    start_time = time.time()

    dataset = biLSTMDataset(input_path, True)

    print("Time for initialisation: ", time.time() - start)

    model = BiLSTM()

    split_and_train(model, dataset, batch_size, learning_rate, num_epochs, device)

    test_dataset = biLSTMDataset("data/curated/test_data")

    test_outputs = test(model, dataset, device, batch_size)