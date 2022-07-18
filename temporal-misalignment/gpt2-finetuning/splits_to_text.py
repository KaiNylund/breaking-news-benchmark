import random
from tqdm import tqdm
import pandas as pd
import nltk

rand = random.Random(123)

def load_year_texts(year, is_train):
  fp = "./WMTdata/decoded_splits_" + str(year) + ".csv"
  year_df = pd.read_csv(fp, delimiter="\t", header=0, index_col=0)
  texts = list(year_df.sentence_split_text)
  print("Total number of texts for year " + str(year) + ": " + str(len(texts)))
  if is_train:
    texts = texts[:(len(texts) // 2)]
  else:
    texts = texts[(len(texts) // 2):]
  rand.shuffle(texts)
  return texts

# Create a train data text file from a single year's csv splits
def splits_to_text(texts, year, max_tokens):
  count = 0
  with open("./WMTdata/token_splits/tokens_" + str(year) + "_" + str(max_tokens) + ".txt", "w") as train_year_file:
    for article in tqdm(texts):
      count += len(nltk.word_tokenize(article))
      if count > max_tokens:
        break

      train_year_file.write("<s>" + article + "</s>")

for year in range(2012, 2022):
  train_year_text = load_year_texts(year, is_train=True)
  test_year_text = load_year_texts(year, is_train=False)
  splits_to_text(train_year_text, year, 100000000)
  splits_to_text(test_year_text, year, 5000000)