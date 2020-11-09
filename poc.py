import gensim.downloader as api
import nltk
from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

path = 'webkb'


def combine_docs() -> list:
    """
    """
    corpus = []
    for p, d, f in os.walk(path):  # path, directory, file
        if f:
            for file in f:
                with open(os.path.join(p, file), 'r', errors='ignore') as infile:
                    contents = infile.read()
                    soup = BeautifulSoup(contents, "html.parser")
                    html_text = "".join(soup.strings)
                    corpus.append(" ".join(html_text.split()))
    return corpus


def process_docs(docs: list) -> list:
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    total_docs = len(docs)
    for i in range(total_docs):
        temp_list = word_tokenize(docs[i].lower())
        docs[i] = " ".join([wordnet_lemmatizer.lemmatize(w) for w in temp_list if w not in stop_words and len(w) > 2])
    return docs


def get_labels() -> list:
    labels = []
    for p, d, f in os.walk(path):
        for file in f:
            labels.append(p.rsplit('/', 1)[1])
    return labels


def write_docs_to_file(docs, labels) -> None:
    if os.path.isfile('docs.xlsx'):
        pass
    else:
        df = pd.DataFrame({'docs': docs, 'labels': labels})
        df.to_excel('docs.xlsx', header=True, engine='xlsxwriter')
    return


def get_features(max_features: int, docs: list) -> list:
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, strip_accents='unicode',
                                 analyzer='word', token_pattern=r'\w{2,}',
                                 ngram_range=(1, 1), max_features=max_features).fit(docs)  # no. of features
    X = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names()
    return X, features


def calculate_purity(labels, clustered) -> float:
    cluster_ids = set(clustered)

    N = len(clustered)
    majority_sum = 0
    for cl in cluster_ids:
        labels_cl = Counter(l for l, c in zip(labels, clustered) if c == cl)
        majority_sum += max(labels_cl.values())

    return majority_sum / N


def tf_idf_purity(X, labels) -> float:
    kmeans = KMeans(n_clusters=5, random_state=0)
    clustered_docs = kmeans.fit_predict(X)

    purity = calculate_purity(labels, clustered_docs)
    return purity


def embedded_representation(total_docs: int,
                            total_features: int,
                            embeds: int,
                            features: list,
                            X: np.ndarray) -> np.ndarray:
    glove_model = api.load('glove-wiki-gigaword-300')
    emb_docs = np.zeros((total_docs, total_features * embeds))
    doc = 0
    for array in X.toarray():
        ft_array_len = len(array)
        for ft in range(ft_array_len):
            if array[ft] > 0 and features[ft] in glove_model:
                emb_docs[doc][ft*300:((ft*300) + 300): 1] = glove_model[features[ft]]
        doc += 1
    return emb_docs


if __name__ == '__main__':
    # docs = combine_docs()
    # docs = process_docs(docs)
    # labels = get_labels()
    # write_docs_to_file(docs, labels)
    # labels = get_labels()
    # print(len(labels))

    max_features = 100
    df = pd.read_excel('docs.xlsx')
    X, features = get_features(max_features, docs=list(df['docs']))
    emb_docs = embedded_representation(
        total_docs=len(df['docs']),
        total_features=max_features,
        embeds=300,
        features=features,
        X=X
    )
    # print(tf_idf_purity(X, list(df['labels'])))
    # print('hello')
    # print('hey')
    kmeans = KMeans(n_clusters=5, random_state=0)
    clustered_docs = kmeans.fit_predict(emb_docs)
    print(calculate_purity(df['labels'], clustered_docs))