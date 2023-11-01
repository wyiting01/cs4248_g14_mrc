"""
The preprocessing function will read in the train and dev json file and extracts the context, question, answer text and answer span (start and end indices in term of character) and save them into different file separated by a tab.

To run the file in root directory, use the following commands

python src/preprocessing.py --file_path "data/raw/train-v1.1.json" --train
python src/preprocessing.py --file_path "data/raw/dev-v1.1.json" --test
"""

import argparse
import json
import requests
import numpy as np

# Adapted from: https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english
url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
american_to_british_dict = requests.get(url).json()

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
        ctxt.append(clean_text(context))
        qns.append(clean_text(question))
        ans.append(clean_text(answer))
        span.append(clean_text(answer_span))
        qns_id.append(clean_text(question_id))
    return ctxt, qns, ans, span, qns_id

# Clean up input text to be more standardised (spelling & word form) and decrease dictionary size.
# This does not expand contractions (isn't -> is not) because answer span will be affected.
# Note: Due to nature of questions & answers, cannot remove all special characters as they are present in them.
def clean_text(text):
    text = text.lower()

    # Clean non-ASCII dashes.
    text = clean_dashes(text)

    # Clean non-ASCII apostrophes and quotation marks.
    text = clean_contractions(text)

    # Changes some American spellings into British spellings. Cannot change all due to answer span.
    text = standardise_spelling(text)

    return text

# Remove non-ASCII dashes.
def clean_dashes(text):
    text = text.replace(u'\u2013', '-')
    return text

# Remove non-ASCII quotes.
def clean_contractions(text):
    specials = [u'\u2018', u'\u2019', u'\u00B4', u'\u0060']
    for s in specials:
        text = text.replace(s, "'")
    return text

# Converts some words from American spelling to British spelling. Note that not all words
# can be converted due to nature of answer spans being fixed.
# e.g. affects z -> s spellings and not 'or' -> 'our' spellings.
def standardise_spelling(text):
    words = text.split()
    for i in range(len(words)):
        if words[i] not in american_to_british_dict.keys():
            continue
        elif len(words[i]) == len(american_to_british_dict[words[i]]):
            # Don't change if the length of words are different as this will affect answer span.
            words[i] = american_to_british_dict[words[i]]
    text = ' '.join(words)
    return text


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