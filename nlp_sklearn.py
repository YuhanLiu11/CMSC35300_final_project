from transformers import AutoTokenizer, AutoModel
import pandas as pd
import csv
from ocsvm import train_ocsvm
import torch
from sklearn import decomposition
from sklearn.svm import OneClassSVM

import argparse
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--gamma', default=0.1, type=float, help='gamma parameter for OCSVM')
args = parser.parse_args()


def read_dataset(csv_path, delimiter):
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        lines = []
        for line in reader:
            lines.append(line)
    return lines

amazon_review_data = read_dataset("../../nlp_dataset/yelp_review_polarity_csv/test.csv", ',')
print(amazon_review_data[0][1])
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inlier_data = []
for i in range(4000):
    try:
        inputs = tokenizer(amazon_review_data[i][1], return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)
        inlier_data.append(outputs.pooler_output.cpu().detach().numpy()[0])
    except:
        continue
print("Number of inlier data: ", len(inlier_data))

outlier_data = []

imdb_review_data = read_dataset("../../nlp_dataset/IMDB_data/test.csv", '\t')
print(imdb_review_data[0])
for i in range(300):
    
    try:
        inputs = tokenizer(imdb_review_data[i][1], return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)

        outlier_data.append(outputs.pooler_output.cpu().detach().numpy()[0])
    except:
        continue

print("Inference done!")
all_data = np.concatenate((np.array(inlier_data), np.array(outlier_data)))
print(all_data.shape)
pca = decomposition.PCA(n_components=10)
pca.fit(all_data)
X = pca.transform(all_data)
clf = OneClassSVM( kernel="rbf", gamma = args.gamma).fit(X)
predicted_classes = clf.predict(X)
outlier_indices = []
for i in range(len(predicted_classes)):
    if predicted_classes[i] < 0:
        print("Outlier index: {}".format(i))
        outlier_indices.append(i)
outlier_indices = np.array(outlier_indices)
tp = len(np.where( outlier_indices > len(inlier_data) )[0])
recall = tp / (len(all_data) - len(inlier_data))
precision = tp / len(outlier_indices)
print(recall)
print(precision)
