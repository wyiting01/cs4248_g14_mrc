"""
The preprocessing function will read in the train and dev json file and extracts the context, question, answer text and answer span (start and end indices in term of character) and save them into different file separated by a tab.

To run the file in root directory, use the following commands

python preprocess/preprocessing.py --file_path "data/raw/train-v1.1.json" --train
python preprocess/preprocessing.py --file_path "data/raw/dev-v1.1.json" --test
"""

import argparse
import json
import numpy as np

def write_to_file(out, path_name):
    with open(path_name, 'w', encoding='utf-8') as out_file:
        out_file.write(out)

# loading training and testing data
def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def preprocess_and_write(dataset):
    examples = []

    # iterate through all articles in dataset
    for articles_id in range(len(dataset['data'])):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = article_paragraphs[pid]['context'] # string

            qas = article_paragraphs[pid]['qas'] # list of questions

            for qn in qas:

                question = qn['question'] # string
                question_id = qn['id'] # question id

                ans_text = qn['answers'][0]['text'] # get the first answer text
                ans_start = qn['answers'][0]['answer_start'] # answer start loc (character count)
                ans_end = ans_start + len(ans_text) # answer end loc (character count) (exclusive)

                examples.append((question_id, question, ans_text, context, ' '.join([str(ans_start), str(ans_end)])))

    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    ctxt = []
    qns = []
    ans = []
    span = []
    qns_id = []

    for i in indices:
        (question_id, question, answer, context, answer_span) = examples[i]
        ctxt.append(context)
        qns.append(question)
        ans.append(answer)
        span.append(answer_span)
        qns_id.append(question_id)
    return ctxt, qns, ans, span, qns_id

def main(args):
    dataset = data_from_json(args.file_path)
    if args.train:
        context, question, answer, answer_span, question_id = preprocess_and_write(dataset)
        write_to_file('\t'.join(context), "data/curated/training_data/context")
        write_to_file('\t'.join(question), "data/curated/training_data/question")
        write_to_file('\t'.join(answer), "data/curated/training_data/answer")
        write_to_file('\t'.join(answer_span), "data/curated/training_data/answer_span")
        write_to_file('\t'.join(question_id), "data/curated/training_data/question_id")
    elif args.test:
        context, question, answer, answer_span, question_id = preprocess_and_write(dataset)
        write_to_file('\t'.join(context), "data/curated/test_data/context")
        write_to_file('\t'.join(question), "data/curated/test_data/question")
        write_to_file('\t'.join(answer), "data/curated/test_data/answer")
        write_to_file('\t'.join(answer_span), "data/curated/test_data/answer_span")
        write_to_file('\t'.join(question_id), "data/curated/test_data/question_id")
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='path to the json file for training or testing')
    parser.add_argument('--train', default=False, action='store_true', help='training data')
    parser.add_argument('--test', default=False, action='store_true', help='testing data')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)