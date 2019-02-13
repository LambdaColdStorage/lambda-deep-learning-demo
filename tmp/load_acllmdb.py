import csv

# myData = [[1, 2, 3], ['Good Morning', 'Good Evening', 'Good Afternoon']]  
# myFile = open('csvexample3.csv', 'w')  
# with myFile:  
#    writer = csv.writer(myFile)
#    writer.writerows(myData)

import os
import csv

DATASET_PATH = "/home/ubuntu/Downloads/aclImdb_v1"
CSV_PATH = "/home/ubuntu/Downloads/aclImdb_v1/test.csv"

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

  # return pos_data + neg_data, pos_labels + neg_labels
  return neg_data + pos_data, neg_labels + pos_labels

data, labels = load_dataset(os.path.join(DATASET_PATH, "aclImdb", "test"))

# save as csv file, seperated by tab
with open(CSV_PATH, 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  for sentence, label in zip(data, labels):
    writer.writerow([sentence, label])
