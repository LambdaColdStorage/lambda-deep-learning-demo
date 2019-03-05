"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Preprocess ACL IMDB dataset
1. Combine raw IMBD files into CSV files
2. Clean text

"""

import argparse
import os
import csv
import re

import pandas as pd
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_directory_data(directory, label):
  data = []
  labels = []
  for file_path in os.listdir(directory):
    with open(os.path.join(directory, file_path), "r") as f:
      s = f.read()
      data.append(s)

      labels.append(label)
  return data, labels


def load_dataset(directory):
  pos_data, pos_labels = load_directory_data(os.path.join(directory, "pos"), 1)
  neg_data, neg_labels = load_directory_data(os.path.join(directory, "neg"), 0)
  return neg_data + pos_data, neg_labels + pos_labels


def ingest_data(csv_path):
    data = pd.read_csv(csv_path,
                       header=None,
                       delimiter='\t',
                       names = ["sentence", "label"])
    data = data[data.label.isnull() == False]
    data['label'] = data['label'].map(int)
    data = data[data['sentence'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    return data


def data_cleaner(text):
  pat_1 = r"(?:\@|https?\://)\S+"
  pat_2 = r'#\w+ ?'
  combined_pat = r'|'.join((pat_1, pat_2))
  www_pat = r'www.[^ ]+'
  html_tag = r'<[^>]+>'
  negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
                  "hadn't":"had not","won't":"will not",
                  "wouldn't":"would not","aren't":"are not",
                  "haven't":"have not", "doesn't":"does not","didn't":"did not",
                   "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
                  "mightn't":"might not",
                  "mustn't":"must not"}
  negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')
  stripped = re.sub(combined_pat, '', text)
  stripped = re.sub(www_pat, '', stripped)
  cleantags = re.sub(html_tag, '', stripped)
  lower_case = cleantags.lower()
  neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
  return neg_handled


def data_remove_punctuation(text):
  text = re.sub("[^a-zA-Z]", " ", text)
  return text


def data_tockenize(text):
  tokenizer = WordPunctTokenizer()
  tokens = tokenizer.tokenize(text)
  return (" ".join(tokens)).strip()


def post_process(data, remove_punctuation, n=1000000):
    data = data.head(n)
    data['sentence'] = data['sentence'].progress_map(data_cleaner)
    if remove_punctuation:
      data['sentence'] = data['sentence'].progress_map(data_remove_punctuation)
    data['sentence'] = data['sentence'].progress_map(data_tockenize)

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

def process_csv(args, split_name):
  raw_csv = os.path.join(args.output_dir, split_name + "_raw.csv")
  clean_csv = os.path.join(args.output_dir, split_name + ".csv")
  data, labels = load_dataset(os.path.join(args.input_dir, split_name))

  # save as csv file, seperated by tab
  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

  with open(raw_csv, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    for sentence, label in zip(data, labels):
      writer.writerow([sentence, label])

  data = ingest_data(raw_csv)

  tqdm.pandas(desc="progress-bar")

  data = post_process(data, args.remove_punctuation)

  data.to_csv(clean_csv, sep='\t', header=False, index=False)


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--input_dir",
                      help="Path for input directory",
                      default="~/demo/data/aclImdb")
  parser.add_argument("--output_dir",
                      help="Path for output directory",
                      default="~/demo/data/IMDB")
  parser.add_argument("--remove_punctuation",
                      help="Flag to remove punnctuation",
                      type=str2bool,
                      default=False)

  args = parser.parse_args()
  args.input_dir = os.path.expanduser(args.input_dir)
  args.output_dir = os.path.expanduser(args.output_dir)

  process_csv(args, "train")
  process_csv(args, "test")


if __name__ == "__main__":
  main()