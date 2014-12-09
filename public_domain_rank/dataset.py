"""Collects features into a single dataset."""

import logging
import os

import numpy as np
import pandas as pd

import public_domain_rank
import public_domain_rank.features as features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset(langlink_max_features, death_decade_start, death_decade_stop):
    """Returns the full dataset

    Limit language link indicators to `langlink_max_features`. Create death
    decade indicators for decades between `death_decade_start` and
    `death_decade_stop`.

    """
    df = features.pageids_canonical().set_index('pageid')
    np.testing.assert_(df.index.is_unique)
    np.testing.assert_(df.notnull().values.all())

    n_records = len(df)

    # merge titles
    titles = features.titles()
    df = pd.merge(df, titles, left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge redirects
    redirects = features.redirects_of_interest()
    df = pd.merge(df, redirects, how='left', left_on='title', right_index=True)
    df['redirect_set'] = df['redirect_set'].fillna(set())
    df['n_redirect'] = df['redirect_set'].apply(len)
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())
    del titles
    del redirects

    # merge topics
    df = pd.merge(df, features.topics(), left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge pagecounts
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.pagecounts(), left_index=True, right_index=True, how='left')
    # because the only column with NaN values is pagecounts, we may fill with 0 safely
    # Note that a pagecount of NaN means a page got 0 or 1 hits during the period studied
    df = df.fillna(0)
    np.testing.assert_((df.pagecount >= 0).all())
    np.testing.assert_equal(n_records, len(df))

    # merge langlinks counts
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.langlink_counts(), left_index=True, right_index=True, how='left')
    # because the only columns with NaN values are dummies, we may fill with 0 safely
    df = df.fillna(0)
    np.testing.assert_(df.notnull().values.all())
    np.testing.assert_equal(n_records, len(df))

    # merge imagelinks
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.imagelink_counts(), left_index=True, right_index=True, how='left')
    df = df.fillna(0)
    np.testing.assert_(df.notnull().values.all())
    np.testing.assert_equal(n_records, len(df))

    # merge article lengths
    df = pd.merge(df, features.article_lengths(df.index), left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge last revision timestamp
    df = pd.merge(df, features.last_revision(df.index), left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))

    # merge article age
    df = pd.merge(df, features.article_age(df.index), left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))

    # merge revisions per day rates
    df = pd.merge(df, features.revisions_per_day(df.index), left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))

    # merge categories (comma-separated list of categories)
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.categories_flat(), left_index=True, right_index=True, how='left')
    # because the only columns with NaN values are dummies, we may fillna safely
    df = df.fillna(set())
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge death decade dummies
    df = pd.merge(df, features.death_decade_dummies(death_decade_start, death_decade_stop), left_index=True, right_index=True, how='left')
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge category dummies of interest
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.category_dummies_of_interest(), left_index=True, right_index=True, how='left')
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge langlink dummies
    np.testing.assert_(df.notnull().values.all())
    df = pd.merge(df, features.langlink_dummies(max_features=langlink_max_features), left_index=True, right_index=True, how='left')
    # because the only columns with NaN values are dummies, we may fillna safely
    df = df.fillna(0)
    np.testing.assert_equal(n_records, len(df))
    np.testing.assert_(df.notnull().values.all())

    # merge online books page data
    df = pd.merge(df, features.digital_editions(), left_index=True, right_index=True, how='left')
    df.obp_digital_editions = df.obp_digital_editions.fillna(0)
    np.testing.assert_equal(n_records, len(df))

    # merge death year (there will be many NaNs)
    df = pd.merge(df, features.death_year(), left_index=True, right_index=True, how='left')
    np.testing.assert_equal(n_records, len(df))

    return df
