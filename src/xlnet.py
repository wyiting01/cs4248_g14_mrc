import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetForQuestionAnswering, XLNetTokenizerFast

class SquadDataset(Dataset):

    def __init__(self, input_path):
        """
        input_path: path that contains all the files - context, question and answer span
        """
        with open(input_path + "/context", encoding='utf-8') as f:
            contexts = f.read().split("\t")
        with open(input_path + "/question", encoding='utf-8') as f:
            questions = f.read().split("\t")
        with open(input_path + "/answer", encoding='utf-8') as f:
            answers = f.read().split("\t")
        with open(input_path + "/answer_span", encoding='utf-8') as f:
            spans = f.read().split("\t")

        self.contexts = [ctx.strip() for ctx in contexts][:10]
        self.questions = [qn.strip() for qn in questions][:10]
        self.answers = [ans.strip() for ans in answers][:10]
        self.spans = [span.strip().split() for span in spans][:10]
        self.start_indices = [int(x[0]) for x in self.spans]
        self.end_indices = [int(x[1]) for x in self.spans]

        self.tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.tokenizer.padding_side = "right"

        self.tokenizer_dict = self.tokenize()
        self.sample_mapping, self.offset_mapping = self.preprocess()

        self.input_ids = self.tokenizer_dict["input_ids"]
        self.token_type_ids = self.tokenizer_dict["token_type_ids"]
        self.attention_mask = self.tokenizer_dict["attention_mask"]


    def tokenize(self, max_length=384, doc_stride=128):
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
        Return the number of instances in the data
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
            "offset_mapping": torch.tensor(self.offset_mapping[i])

        }
        return item_dict
    
def train(model, dataset, batch_size=16, learning_rate=5e-5, num_epoch=10, device='cpu', model_path=None):

    data_loader = DataLoader(dataset, batch_size=batch_size)#, shuffle=True)
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

def test_one(model, dataset, n_best_size=20, max_answer_length=30, device='cpu'):
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    one_batch = next(iter(test_dataloader))

    model.eval()
    with torch.no_grad():
        input_ids = one_batch["input_ids"].to(device)
        attention_mask = one_batch["attention_mask"].to(device)
        start = one_batch["start_positions"].to(device)
        end = one_batch["end_positions"].to(device)
        og_index = one_batch["og_indices"]
        context = one_batch["og_contexts"][0]
        offset_mapping = one_batch["offset_mapping"][0]

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = output.start_top_log_probs[0].cpu().detach().numpy()
        end_logits = output.end_top_log_probs[0].cpu().detach().numpy()
        start_indexes = output.start_top_index[0].cpu().numpy()
        end_indexes = output.end_top_index[0].unique_consecutive().cpu().numpy()

        valid_answers=[]
        for i, start_index in enumerate(start_indexes):
            for j, end_index in enumerate(end_indexes):
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[i] + end_logits[j],
                            "text": context[start_char: end_char]
                        }
                    )
        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
    return valid_answers

def test(model, dataset, n_best_size=20, max_answer_length=30, device='cpu'):
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    pred = {}

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

            for i in range(len(input_ids)):

                start_logits = output.start_top_log_probs[i].cpu().detach().numpy()
                end_logits = output.end_top_log_probs[i].cpu().detach().numpy()
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                start_top_indexes = output.start_top_index
                end_top_indexes = output.end_top_index


                offsets = offset_mapping[i]
                ctxt = context[i]

                valid_answers = []
                for start in start_indexes:
                    for end in end_indexes:
                        start_index = start_top_indexes[start]
                        end_index = end_top_indexes[end]
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offsets)
                            or end_index >= len(offsets)
                            or offsets[start_index] is None
                            or offsets[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                            start_char = offsets[start_index][0]
                            end_char = offsets[end_index][1]
                            valid_answers.append(
                                {
                                    "score": start_logits[start] + end_logits[end],
                                    "text": ctxt[start_char: end_char]
                                }
                            )

                valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
                if len(valid_answers) == 0:
                    pred[answer[i]] = ""
                else:
                    pred[answer[i]] = valid_answers[0]['text']

    return pred

def main(args):
    train_path, test_path, model_path, output_path = args.train_path, args.test_path, args.model_path, args.output_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    squad_train = SquadDataset(train_path)
    model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased').to(device)

    # train the model
    train(model=model, dataset=squad_train, num_epoch=1, batch_size=16, device=device, model_path = model_path)

    # test the model (to be modified from trg data to test data)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    squad_test = SquadDataset(test_path)
    test_output = test(model, dataset=squad_test, device=device)
    print(test_output)

    # ls = []
    # for key, val in test_output.items():
    #     ls.append(key + "\t" + val)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(ls))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path to the text file')
    parser.add_argument('--test_path', help='path to the text file')
    parser.add_argument('--model_path', help='path to save trained model')
    parser.add_argument('--output_path', help='path to model_prediction')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)