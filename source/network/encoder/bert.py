import numpy as np


def encode(sentences, vocab, max_seq_length):

  def run(sentence):

    # Add special tokens to the sentence
    if len(sentence) > max_seq_length - 2:
      sentence = sentence[0:(max_seq_length - 2)]
    tokens = []
    tokens.append("[CLS]")
    for token in sentence:
      tokens.append(token)
    tokens.append("[SEP]")

    # Encode sentence by vocabulary id
    encode_sentence = [vocab[w] for w in tokens if w in vocab]

    mask = [1] * len(encode_sentence)
    while len(encode_sentence) < max_seq_length:
      encode_sentence.append(0)
      mask.append(0)

    return np.array(encode_sentence, dtype="int64"), np.array(mask, dtype="int64")

  encode_sentences, masks = zip(*[run (s) for s in sentences])
  return encode_sentences, masks
