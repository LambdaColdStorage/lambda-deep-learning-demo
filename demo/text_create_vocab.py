"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================
"""

"""
python demo/text_create_vocab.py \
--dataset_name=CoLA \
--data_file=/home/ubuntu/demo/data/CoLA/in_domain_train.tsv \
--vocab_file=/home/ubuntu/demo/data/CoLA/vocab.pkl
"""

import argparse
import os
import csv
import re
from collections import Counter
import operator
import pickle


def loadSentences_CoLA(file_name):
  sentences = []
  for file in file_name:
    dirname = os.path.dirname(file)
    with open(file) as f:
      parsed = csv.reader(f, delimiter="\t")
      for row in parsed:        
        sentences.append(re.findall(r"[\w']+|[.,!?;]", row[3]))
        
  return sentences


def buildVocab(sentences):
  list_words = [w for l in sentences for w in l]
  counter = Counter(list_words)
  word_cnt = sorted(counter.items(),
                    key=operator.itemgetter(1), reverse=True)
  words = [x[0] for x in word_cnt]
  words2idx = { w : i for i, w in enumerate(words)}
  return words2idx, words



def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dataset_name",
                      type=str,
                      choices=["CoLA"],
                      help="The iput dataset name",
                      default="train")
  parser.add_argument("--data_file",
                      type=str,
                      help="The text file to extract vocabulary",
                      default="")
  parser.add_argument("--vocab_file",
                      type=str,
                      help="Path to save the output vacobulary",
                      default="")
  args = parser.parse_args()
  
  if args.dataset_name == 'CoLA':
    sentences = loadSentences_CoLA(args.data_file.split(","))
  else:
    print("Not implemented")

  words2idx, words = buildVocab(sentences)
  with open(args.vocab_file, 'wb') as f:
    pickle.dump([words2idx, words], f)

if __name__ == "__main__":
  main()