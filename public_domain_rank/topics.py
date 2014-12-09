"""Routines for extracting topics from topic model fit."""
import os

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text

import public_domain_rank


def theta():
    """Returns an estimate of the document-topic distributions."""
    datastem = os.path.expanduser(public_domain_rank.config['data']['topics-datastem'])
    fitstem = os.path.expanduser(public_domain_rank.config['data']['topics-fitstem'])
    docnames_fn = '{}.documents'.format(datastem)
    docnames = tuple(os.path.splitext(os.path.basename(n))[0] for n in open(docnames_fn).read().split())
    pageids = tuple(int(docname) for docname in docnames)
    with open('{}.ndt'.format(fitstem)) as f:
        num_docs = int(f.readline().strip())
        num_topics = int(f.readline().strip())
        np.testing.assert_equal(num_docs, len(docnames))
        f.readline()  # discard third line
        ndt = np.zeros((num_docs, num_topics))
        for line in f:
            d, t, cnt = (int(elem) for elem in line.split())
            ndt[d, t] = cnt
    theta = ndt / ndt.sum(axis=1, keepdims=True)
    np.testing.assert_equal(len(docnames), len(theta))
    topic_names = ['topic{:03d}'.format(i) for i in range(num_topics)]
    df = pd.DataFrame(theta, columns=topic_names, index=pageids)
    np.testing.assert_(df.index.is_unique)
    df.index.name = 'pageid'
    return df


def phi():
    """Returns an estimate of the topic-word distributions."""
    datastem = os.path.expanduser(public_domain_rank.config['data']['topics-datastem'])
    fitstem = os.path.expanduser(public_domain_rank.config['data']['topics-fitstem'])
    vocab_fn = '{}.tokens'.format(datastem)
    vocab = tuple(open(vocab_fn).read().split())

    with open('{}.nwt'.format(fitstem)) as f:
        vocab_size = int(f.readline().strip())
        num_topics = int(f.readline().strip())
        f.readline()  # discard third line
        np.testing.assert_equal(vocab_size, len(vocab))
        ntw = np.zeros((num_topics, vocab_size))
        for line in f:
            w, t, cnt = (int(elem) for elem in line.split())
            ntw[t, w] = cnt
    phi = ntw / ntw.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(phi.sum(axis=1), 1)
    topic_names = ['topic{:03d}'.format(i) for i in range(num_topics)]
    df = pd.DataFrame(phi, columns=vocab, index=topic_names)
    np.testing.assert_(df.index.is_unique)

    return df


def ntw():
    """Returns raw auxiliary variable counts related to topic-word distributions."""
    datastem = os.path.expanduser(public_domain_rank.config['data']['topics-datastem'])
    fitstem = os.path.expanduser(public_domain_rank.config['data']['topics-fitstem'])
    vocab_fn = '{}.tokens'.format(datastem)
    vocab = tuple(open(vocab_fn).read().split())

    with open('{}.nwt'.format(fitstem)) as f:
        vocab_size = int(f.readline().strip())
        num_topics = int(f.readline().strip())
        f.readline()  # discard third line
        np.testing.assert_equal(vocab_size, len(vocab))
        ntw = np.zeros((num_topics, vocab_size))
        for line in f:
            w, t, cnt = (int(elem) for elem in line.split())
            ntw[t, w] = cnt
    topic_names = ['topic{:03d}'.format(i) for i in range(num_topics)]
    df = pd.DataFrame(ntw, columns=vocab, index=topic_names)
    np.testing.assert_(df.index.is_unique)

    return df


def top_words(phi, n=20):
    """Get top `n` words for each topic."""
    top_words = []
    for i, row in phi.iterrows():
        row.sort(ascending=False)
        top_words.append(tuple(row[:n].index))
    return top_words


def top_words_tfidf(ntw, n=20, sublinear_tf=False):
    """Get top `n` words for each topic using TF-IDF."""
    top_words = []
    dtm = sklearn.feature_extraction.text.TfidfTransformer(sublinear_tf=sublinear_tf).fit_transform(ntw.values).toarray()
    vocab = ntw.columns
    for row in dtm:
        top_words.append(tuple(vocab[row.argsort()[::-1][:n]]))
    return top_words
