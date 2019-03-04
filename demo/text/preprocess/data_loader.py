import six
import re

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
