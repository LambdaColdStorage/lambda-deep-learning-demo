import six
import re
import pandas as pd


def char_basic(input_file):
  data = []

  with open(input_file, 'rb') as f:
    d = f.read()
  if six.PY2:
    d = bytearray(d)
    data.extend([chr(c) for c in d])

  return data

def word_basic(input_file):
  data = []

  with open(input_file, 'rb') as f:
    d = f.read()
    d = re.findall(r"[\w']+|[:.,!?;\n]", d)
    data.extend(d)

  return data

def imdb_loader(input_file):
  data = pd.read_csv(input_file,
                     header=None,
                     delimiter='\t',
                     names = ["sentence", "label"])
  data = data[data.label.isnull() == False]
  data['label'] = data['label'].map(int)
  data = data[data['sentence'].isnull() == False]
  data.reset_index(inplace=True)
  data.drop('index', axis=1, inplace=True)
  return data