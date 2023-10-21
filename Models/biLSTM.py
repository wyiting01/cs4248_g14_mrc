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
from torch.utils.data import DataLoader

# Randomly initialised hyperparameters. To be put into a grid search.
embed_size = 360
max_features = 100000
maxQnLen = 80
batch_size = 64
num_epochs = 10
hidden_size = 128
x_train, y_train, x_test, y_test = 0
seed = 0

# Ensure no randomisation for every iteration of run.
def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def extractData(file_path1, file_path2):
    newDataset = []
    with open(file_path1, "r") as file1, open(file_path2, "r") as file2:
        for line1, line2 in zip(file1, file2):
            newDataset.append([line1.strip(), line2.strip()])
    return newDataset

def initialiseDataSets():
    # Initialise training set.
    passage = "../data/training_data/context"
    questions = "../data/training_data/question"
    answer_spans = "../data/training_data/answer_span"


    train_dataset = extractData(questions, answers)

    X_train, Y_train = train_dataset

    # Initialise test set.
    passage = "../data/test_data/context"
    questions = "../data/test_data/question"
    answer_spans = "../data/test_data/answer_span"

    test_dataset = extractData(questions, answers)

    X_test, Y_test = test_dataset

    question, answer = train_dataset[0]

    print("Example of original training question & answer:", question, answer)

    # Tokenise sentences.
    tokeniser = Tokenizer(num_words=max_features)
    # Since input is text.
    tokeniser.fit_on_texts(list(X_train))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad the sentences.
    X_train = pad_sequences(X_train, maxlen=maxQnLen)
    X_test = pad_sequences(X_test, maxlen=maxQnLen)

    # Reshuffle the data.   
    np.random.seed(seed)

    training_idx = np.random.permutation(len(X_train))
    X_train = X_train[training_idx]
    Y_train = Y_train[training_idx]

    return X_train, Y_train, X_test, Y_test, tokenizer.word_index

# Pytorch version: Adapted from https://www.kaggle.com/code/mlwhiz/bilstm-pytorch-and-keras

class Dataset(dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

class BiLSTM(nn.Module):
    
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , self.hidden_size)
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
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

def split_and_train(x_train, y_train, x_test, batch_size, learning_rate, num_epoch, device='cpu'):
    seed_all()

    all_avg_loss = []

    all_avg_val_loss = []

    train_predict = np.zeroes((len(x_train)))

    test_predict = np.zeroes((len(x_test)))

    # Combines a Sigmoid layer and the BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    optimiser = optim.Adam(model.parameters, lr=learning_rate)

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
        for epoch in range(num_epoch):

            start_time = time.time()
            model.train()
            avg_loss = 0.0
            for step, data in enumerate(dataloader, 0):
                texts, labels = data[0].to(device), data[1].to(device)

                outputs = model(texts)

                loss = criterion(outputs)

                loss.backward()

                optimiser.step()

                avg_loss = loss.item() / len(dataloader)

            model.eval()


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    seed_all()

    start_time = time.time()

    x_train, y_train, x_test, y_test, word_index = initialiseDataSets()

    print("Time for initialisation: ", time.time() - start)

    split_and_train(x_train, y_train, x_test, batch_size, learning_rate, num_epochs)