from concurrent.futures import process
from spacy.lang.en import English
import spacy
from spacy.pipeline import EntityRecognizer
from spacy.pipeline.ner import DEFAULT_NER_MODEL
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')
#nlp = English()
#ner = EntityRecognizer(nlp.vocab, model)

#ner = nlp.add_pipe("ner")
#ner.initialize(lambda: [], nlp=nlp)

#with open("./WMTdata/decoded_splits_" + str(2012) + ".csv", "r") as t:
#  print(t.readline())

for year in range(2012, 2022):
  year_csv = pd.read_csv("./WMTdata/decoded_splits_" + str(year) + ".csv", sep="\t", nrows=1000)
  split_texts = list(year_csv["sentence_split_text"])

  entity_counts = defaultdict(int)

  for text in tqdm(split_texts):
    text_doc = nlp(text)

    for ent in text_doc.ents:
      entity_counts[ent.label_] += 1

    for token in text_doc:
      entity_counts[token.pos_] += 1
      entity_counts[token.tag_] += 1

  print(entity_counts)
  np.save(str(year) + "_ent_counts", entity_counts)
  #with open("./WMTdata/text_" + str(year) + "_10000.txt", "r") as ex_doc_file:
  #  year_text = ex_doc_file.read()[:100000]
  #  year_text = nlp(year_text)
  plt.bar(entity_counts.keys(), entity_counts.values())
  plt.show()

'''
ex_text = ""
with open("./WMTdata/text_2012_1000.txt", "r") as ex_doc_file:
  ex_text = ex_doc_file.read()[:100000]

ex_doc = nlp(ex_text)
#print(ex_doc)

for ent in ex_doc.ents:
  print(ent.text, ent.label_)

for token in ex_doc:
  print(token.pos_, token.tag_)
#scores = ner.predict([ex_doc])
#print(scores)
'''