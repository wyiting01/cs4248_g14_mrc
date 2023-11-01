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
    parser.add_argument("--answer", help="Path to the question input file")
    parser.add_argument("--question_id", help = "Path to Question ID file")
    parser.add_argument("--output_file", help="Path to the output file")

    args = parser.parse_args()
    answers = read_file(args.answer)
    qid = read_file(args.question_id)

    answers = re.split("\t", answers)
    qid = re.split("\t", qid)

    test_answers = {}

    for i in range(len(answers)):
        # print(i)
        test_answers[qid[i]] = answers[i]
        
    
    with open(args.output_file, 'w+') as f:
            json.dump(test_answers, f)

if __name__ == "__main__":
    main()