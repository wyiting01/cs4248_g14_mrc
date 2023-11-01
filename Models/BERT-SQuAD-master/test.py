<<<<<<< HEAD
"""
Script to run testing for BERT

command to run script: python3 test.py *path to question file* *path to context file*

it will return answers to the test data given.
"""
import argparse
import re
from bert import QA
import json

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
            return file_contents
    except FileNotFoundError:
        return f"File not found at '{file_path}'"
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    parser = argparse.ArgumentParser(description="Read the contents of question and context files")
    parser.add_argument("--question_input", help="Path to the question input file")
    parser.add_argument("--question_id", help = "Path to Question ID file")
    parser.add_argument("--context_input", help="Path to the context input file")
    parser.add_argument("--output_file", help="Path to the output file")

    args = parser.parse_args()
    question_contents = read_file(args.question_input)
    context_contents = read_file(args.context_input)
    qid = read_file(args.question_id)

    questions = re.split("\t", question_contents)
    qid = re.split("\t", qid)
    contexts = re.split("\t", context_contents)
    print("Total:" + str(len(questions)))

    answers = {}

    bert = QA('model')

    for i in range(len(questions)):
        print(i)
        curr_ans = bert.predict(contexts[i], questions[i])
        answers[qid[i]] = curr_ans['answer']
    
    with open(args.output_file, 'w+') as f:
            json.dump(answers, f)

if __name__ == "__main__":
    main()

#10570
=======
"""
Script to run testing for BERT

command to run script: python3 test.py *path to question file* *path to context file*

it will return answers to the test data given.
"""
import argparse
import re
from bert import QA

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
            return file_contents
    except FileNotFoundError:
        return f"File not found at '{file_path}'"
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    parser = argparse.ArgumentParser(description="Read the contents of question and context files")
    parser.add_argument("question_input", help="Path to the question input file")
    parser.add_argument("context_input", help="Path to the context input file")

    args = parser.parse_args()
    question_contents = read_file(args.question_input)
    context_contents = read_file(args.context_input)

    questions = re.split("\t", question_contents)
    contexts = re.split("\t", context_contents)

    answers = []
    answers_span = []

    bert = QA('model')

    for i in range(len(questions)):
        curr_ans = bert.predict(contexts[i], questions[i])
        answers.append(curr_ans['answer'])
        answers_span.append((curr_ans['start'], curr_ans['end']))
    return answers, answers_span

if __name__ == "__main__":
    main()
>>>>>>> Revert preprocessing and added k fold validations
