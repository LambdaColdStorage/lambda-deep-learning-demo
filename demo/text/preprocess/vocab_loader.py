import pickle
import numpy as np


def load(vocab_file, vocab_format, top_k):
  if vocab_format == "pickle":
    f = open(vocab_file,'r')  
    items = pickle.load(f)
    if top_k > 0 and len(items) > top_k:
      items = items[:top_k]
    vocab = { w : i for i, w in enumerate(items)}
    embd = []
  elif vocab_format == "txt":
    items = []
    embd = []
    file = open(vocab_file,'r')
    count = 0
    for line in file.readlines():
        row = line.strip().split(' ')
        items.append(row[0])
        
        if len(row) > 1:
          embd.append(row[1:])

        count += 1
        if count == top_k:
          break
    file.close()
    vocab = { w : i for i, w in enumerate(items)}
    if embd:
      embd = np.asarray(embd).astype(np.float32)

  return vocab, items, embd  
