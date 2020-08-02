import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'Starspace_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    '''embeddings_df = pd.read_csv(embeddings_path,sep='\t')
    embedding_keys = embeddings_df.values[:,0]
    embedding_values = embeddings_df.values[:,1:].astype('float32')
    embeddings  = {embedding_keys[i]:embedding_values[i] for i in range(embeddings_df.shape[0])}
    embeddings_dim = embeddings[list(embeddings.keys())[0]].shape
    return embeddings,embeddings_dim'''
    embeddings = {}
    dim=0
    for line in open(embeddings_path):
        word,*vec = line.strip().split()
        dim = len(vec)
        embeddings[word] = np.array(vec,dtype=np.float32)
    return embeddings,dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    question = list(question.split())
    result = np.zeros((dim))
    l = 0
    for word in question:
      if word in embeddings:
        l += 1
        result += embeddings[word]
    if l>0:
      result = result/l
    return result


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
