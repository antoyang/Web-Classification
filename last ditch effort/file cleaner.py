"""Cleaning the text files to make them readable"""

import codecs
import re
import nltk
import pickle

file_type = 'test'
file_name = file_type + '.csv'
relevant_idx = list()
with open("../data/" + file_name, 'r') as f:
    for line in f:
        relevant_idx.append(line.split(',')[0].replace("\n", ""))

docs_sentences = dict()
for idx in relevant_idx:
    with codecs.open("../data/text/text/" + idx, "r", encoding='utf-8', errors='ignore') as f:
        content = f.read()
        content = re.sub('[^A-za-zÀ-ÿ0-9\s,&\'\-\.\!\?\n]+', '', content)
        sentences = nltk.tokenize.sent_tokenize(content, language="french")
        current_doc_sentences = list()
        for s in sentences:
            if len(s.split(' ')) > 5 and len(s.split('\n')) < 30:
                s = s.split(' ')
                s = ' '.join(s[:500])
                current_doc_sentences.append(s)
        docs_sentences[idx] = current_doc_sentences

print(docs_sentences[0])
with open(file_type + "_sentences.pkl", "wb") as f:
    pickle.dump(docs_sentences, f)
