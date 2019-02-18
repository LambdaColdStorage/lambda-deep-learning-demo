import numpy as np


def basic(sentences, vocab, max_seq_length):

  def encode(sentence):
    encode_sentence = [vocab[w] for w in sentence if w in vocab]
    encode_sentence = encode_sentence[0:max_seq_length]

    mask = [1] * len(encode_sentence)

    while len(encode_sentence) < max_seq_length:
      encode_sentence.append(0)
      mask.append(0)

    return np.array(encode_sentence, dtype="int32"), np.array(mask, dtype="int32")

  encode_sentences, encode_masks = zip(*[encode (s) for s in sentences])
  return encode_sentences, encode_masks

def glove(sentences, vocab):
  pass  


def bert(sentences, vocab):
  pass