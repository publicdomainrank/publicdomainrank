"""Prefetch pagecounts of interest. This step takes more than a hour."""

import argparse
import collections
import itertools
import logging
import os

import numpy as np
import pandas as pd

import public_domain_rank
import public_domain_rank.features as features


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('public_domain_rank')

_pagecounts_store_fn = os.path.expanduser(public_domain_rank.config['data']['pagecounts'])
_pagecounts_store = None  # lazy load
PAGECOUNTS_NUM_DAYS = 60  # page count total is drawn from 60 random days


def _redirects_of_interest(titles_of_interest):
    titles = features.titles()
    redirects = features.redirects()
    # restrict redirects to universe and add pageid
    redirects = redirects[redirects.redirect_to.isin(titles_of_interest)].reset_index(drop=True)
    redirects = pd.merge(redirects, titles, how='left', left_on='redirect_from', right_index=True)
    del redirects['redirect_from']  # old redirect_from, was pageid
    redirects.rename(columns={'title': 'redirect_from'}, inplace=True)
    redirects = redirects.set_index('redirect_to')

    # remove NaN; these correspond to pages that have no title in the dump (likely deleted?)
    redirects = redirects[pd.notnull(redirects.redirect_from)]

    # now the redirects DataFrame is in a usable form
    redirects = redirects.groupby(redirects.index)['redirect_from'].apply(set)
    redirects.name = 'redirect_set'
    return pd.DataFrame(redirects)


def pagecounts_for_titles(titles):
    """Returns page view counts for page with titles `titles`.

    Returns a dictionary with titles as keys.
    """
    global _pagecounts_store
    if not isinstance(titles, collections.abc.Iterable):
        raise ValueError("Expected an iterable.")
    if _pagecounts_store is None:
        _pagecounts_store = pd.HDFStore(_pagecounts_store_fn, 'r')
    # HDF5 querying was changed recently in Pandas, you really do refer to
    # the local variable in the string in Pandas 0.14.
    titles_set = set(titles)  # noqa
    results = _pagecounts_store.select('df', 'index in titles_set')
    return (results['hits'] / PAGECOUNTS_NUM_DAYS).to_dict()


def dataset():
    """Returns partial dataset with pagecounts."""
    df = features.pageids_canonical().set_index('pageid')
    np.testing.assert_(df.index.is_unique)
    n_records = len(df)

    # merge titles
    titles = features.titles()
    df = pd.merge(df, titles, left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))

    # merge redirects
    redirects = _redirects_of_interest(df.title)
    df = pd.merge(df, redirects, how='left', left_on='title', right_index=True)
    df['redirect_set'] = df['redirect_set'].fillna(set())
    df['n_redirect'] = df['redirect_set'].apply(len)
    np.testing.assert_equal(n_records, len(df))
    del titles
    del redirects

    # gather pagecounts for all titles associated with a page
    # this code is awkward because we need to avoid making many lookups in a
    # very large HDF5 file. We make just one query and then rearrange results.
    titles_and_redirect_titles_of_interest = set(df.title.tolist())
    for s in df.redirect_set:
        titles_and_redirect_titles_of_interest.update(s)

    # to avoid memory errors we do this in stages
    def chunks(iterable, size):
        it = iter(iterable)
        item = list(itertools.islice(it, size))
        while item:
            yield item
            item = list(itertools.islice(it, size))

    chunk_size = 30
    titles_chunks = chunks(titles_and_redirect_titles_of_interest, chunk_size)
    pagecounts = dict()
    logger.info("Retrieving pagecounts. This may take several hours.")
    for titles in titles_chunks:
        pagecounts_partial = pagecounts_for_titles(titles)
        pagecounts.update(pagecounts_partial)
        if len(pagecounts) % 10000 <= chunk_size:
            logging.info("Retrieving pagecount {} of {}".format(len(pagecounts), len(titles_and_redirect_titles_of_interest)))

    pagecounts_ordered = []
    for i, (index, row) in enumerate(df.iterrows()):
        count = pagecounts.get(row['title'], 0)
        for redirect_title in row['redirect_set']:
            count += pagecounts.get(redirect_title, 0)
        pagecounts_ordered.append(count)
    df['pagecount'] = pagecounts_ordered
    return df[['title', 'pagecount']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save pagecounts of interest to csv')
    parser.add_argument('output', type=str, help='Output CSV')
    args = parser.parse_args()
    df = dataset()
    df.to_csv(args.output)
