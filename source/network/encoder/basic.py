import numpy as np


def encode(sentences, vocab, max_seq_length):

  def run(sentence):
    encode_sentence = [vocab[w] for w in sentence if w in vocab]
    if max_seq_length > 0:
      encode_sentence = encode_sentence[0:max_seq_length]

    mask = [1] * len(encode_sentence)

    if max_seq_length > 0:
      while len(encode_sentence) < max_seq_length:
        encode_sentence.append(0)
        mask.append(0)

    return np.array(encode_sentence, dtype="int32"), np.array(mask, dtype="int32")

  encode_sentences, encode_masks = zip(*[run (s) for s in sentences])
  return encode_sentences, encode_masks

