import pandas as pd
import re
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer

raw_csv = "/home/ubuntu/Downloads/aclImdb_v1/test.csv"
clean_csv = "/home/ubuntu/Downloads/aclImdb_v1/test_clean.csv"

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
      letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
      tokens = tokenizer.tokenize(letters_only)
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

data = ingest_data(raw_csv)

tqdm.pandas(desc="progress-bar")

data = post_process(data)

data.to_csv(clean_csv, sep='\t', header=False, index=False)

# visualize_wordcloud(data)

# for index, row in data.iterrows():
#   print(type(row['sentence']))
#   print(row['sentence'])
#   print(type(row['label']))
#   break
