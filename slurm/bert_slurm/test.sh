#!/bin/sh

python test.py \
    --test\
	--question_input "../data/curated/test_data/question"\
	--context_input "../data/curated/test_data/context"\
    --output_file "./pred.json"