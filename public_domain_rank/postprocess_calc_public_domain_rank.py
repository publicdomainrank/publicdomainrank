"""
After having fit the public domain rank model, calculate public domain rank
and public domain score for all pages and save data to disk.
"""
import os
import pickle
import sys

import pandas as pd
import numpy as np
import scipy.stats

from public_domain_rank import calc_public_domain_rank

scaler = calc_public_domain_rank.scaler  # scaled on training data
feature_names = calc_public_domain_rank.feature_names

fit_fn = sys.argv[1]

##############################################################################
# Calculate Public Domain Rank and Public Domain Score
##############################################################################
if __name__ == "__main__":
    extr = pickle.load(open(fit_fn, 'rb'))
    beta = extr['beta']
    beta = beta[0:5000, :]  # trim to fit in memory

    df = calc_public_domain_rank.df
    X = scaler.transform(df[feature_names])

    xbeta = np.dot(X, beta.T)
    xbeta_mean = np.mean(xbeta, axis=1)
    np.testing.assert_equal(xbeta_mean.shape, (len(df),))

    df_display = df[['title']]
    df_display['xbeta_mean'] = xbeta_mean
    # free up some memory
    del df
    del X

    def ranking(v):
        return np.argsort(np.argsort(v)[::-1]) + 1

    rank_sim = np.apply_along_axis(ranking, 0, xbeta)
    rank_expected = np.mean(rank_sim, axis=1)
    del rank_sim
    np.testing.assert_equal(rank_expected.shape, (len(df_display),))

    # final ranking, in terms of expected rank
    df_display['rank'] = ranking(-1 * rank_expected)

    # expected quantile
    ecdf_bins = scipy.stats.mstats.mquantiles(xbeta.ravel(), prob=np.linspace(0, 1, 1000))
    quantile_sim = np.apply_along_axis(lambda row: np.digitize(row, ecdf_bins), 1, xbeta) / 10
    quantile_expected = np.mean(quantile_sim, axis=1)
    np.testing.assert_equal(quantile_expected.shape, (len(df_display),))
    df_display['score'] = quantile_expected

    results_fn = '{}.csv'.format(os.path.splitext(fit_fn)[0])
    df_display.to_csv(results_fn)
