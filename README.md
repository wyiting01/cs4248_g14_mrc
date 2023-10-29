CS4248 G14 Machine Reading Comprehension on SQuAD

Base Models
1) BERT
   - File bert.py
   - Model: BERT-SQuAD
     {"exact": 57.79564806054872,"f1": 72.18288133527203}

2) XLNet
   - File: xlnet.py
   - Model: XLNetForQuestionAnswering
     {"exact": 75.34531693472091, "f1": 84.06374401013184}
   - Training: python xlnet.py --train --data_path "../data/curated/training_data" --model_path "./xlnet.pt"
   - Testing: python xlnet.py --test --data_path "../data/curated/test_data" --model_path "./xlnet.pt" --output_path "./pred.json"

4) biLSTM
