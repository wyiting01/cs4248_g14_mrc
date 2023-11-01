"""
This script runs 1 iteration of the bert model.

To run file: python3 single_test.py

doc: Context provided. Change context for other testing

q: Question provided. Change question for other testing.

output: dictionary containing the answer(text), answer(span), confidence (probability) and the document (context)
"""
from bert import QA

model = QA('model')

doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."
q = 'When did Victoria enact its constitution?'


answer = model.predict(doc,q)

print(answer['start'])

print(answer.keys())