import pandas as pd
import re
from collections import Counter
import operator


from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer


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

  tokenizer = WordPunctTokenizer()

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

  try:
      stripped = re.sub(combined_pat, '', text)
      stripped = re.sub(www_pat, '', stripped)
      cleantags = re.sub(html_tag, '', stripped)
      lower_case = cleantags.lower()
      neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
      
      if remove_punctuation:
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        tokens = tokenizer.tokenize(letters_only)
      else:
        tokens = tokenizer.tokenize(neg_handled)

      return (" ".join(tokens)).strip()
  except:
      return 'NC'


def post_process(data, n=1000000):
    data = data.head(n)
    data['sentence'] = data['sentence'].progress_map(data_cleaner)  
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def visualize_wordcloud(data):
  import matplotlib.pyplot as plt
  plt.style.use('fivethirtyeight')

  from wordcloud import WordCloud, STOPWORDS

  neg_tweets = data[data.label == 0]
  neg_string = []
  for t in neg_tweets.sentence:
      neg_string.append(t)
  neg_string = pd.Series(neg_string).str.cat(sep=' ')
  from wordcloud import WordCloud

  wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='white').generate(neg_string)
  plt.figure(figsize=(12,10))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.show()


def buildVocab(sentences, top_k):
  list_words = [w for l in sentences for w in l.split(" ")]
  counter = Counter(list_words)
  word_cnt = sorted(counter.items(),
                    key=operator.itemgetter(1), reverse=True)
  top_k = min(top_k, len(word_cnt))
  words = [x[0] for x in word_cnt[:top_k]]
  words2idx = { w : i for i, w in enumerate(words)}
  return words2idx, words


names_raw_csv = ["/home/ubuntu/demo/data/IMDB/train.csv", "/home/ubuntu/demo/data/IMDB/test.csv"]
names_clean_csv = ["/home/ubuntu/demo/data/IMDB/train_clean.csv", "/home/ubuntu/demo/data/IMDB/test_clean.csv"]
create_vocab = [True, False]
names_vocab = ["/home/ubuntu/demo/data/IMDB/vocab_basic.txt", ""]
flags_visualization = [False, False]
remove_punctuation = False
top_k = 100000

for name_raw_csv, name_clean_csv, flag_vocab, name_vocab, flag_visualization in zip(names_raw_csv, names_clean_csv, create_vocab, names_vocab, flags_visualization):
  data = ingest_data(name_raw_csv)

  tqdm.pandas(desc="progress-bar")

  data = post_process(data)

  data.to_csv(name_clean_csv, sep='\t', header=False, index=False)

  if flag_vocab:
    words2idx, words = buildVocab(data['sentence'], top_k)
    f = open(name_vocab, "w")
    for w in words:
      f.write("%s\n" % w)
    f.close()

  if flag_visualization:
    visualize_wordcloud(data)
