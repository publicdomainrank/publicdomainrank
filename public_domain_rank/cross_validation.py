import collections
import os
import sys

import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_validation
import sklearn.linear_model as linear_model

from public_domain_rank import calc_public_domain_rank

y_train = calc_public_domain_rank.y_train.copy()
X_train = calc_public_domain_rank.X_train.copy()
feature_names = np.array(calc_public_domain_rank.feature_names)
BASE_DIR = calc_public_domain_rank.BASE_DIR
DEBUG = calc_public_domain_rank.DEBUG
POINT_ESTIMATE = calc_public_domain_rank.POINT_ESTIMATE
NUM_ITER = calc_public_domain_rank.NUM_ITER

TEST_SIZE = 1. / 2
NUM_SPLITS = 1  # run different random seeds for parallelism

# cross-validation internal
try:
    RANDOM_STATE = int(sys.argv[1])
except IndexError:
    print("Specify a random seed as an argument to the script")
print("RANDOM_STATE: ", RANDOM_STATE)


##############################################################################
# helper functions
##############################################################################

def log_loss(proba, y):
    """Vectorized log loss

    y are 0, 1 labels
    proba are predicted probabilities

    """
    np.testing.assert_array_less(0 - 1e5, y)
    np.testing.assert_array_less(y, 1 + 1e5)
    np.testing.assert_array_less(0 - 1e5, proba)
    np.testing.assert_array_less(proba, 1 + 1e5)
    return -1 * np.sum(np.log(y * proba + (1 - y) * (1 - proba)))


def invlogit(xbeta):
    return 1/(1 + np.exp(-1 * xbeta))


def cross_validation_log_loss_baseline(y):
    losses = []
    sss = cross_validation.StratifiedShuffleSplit(y, NUM_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in sss:
        y_train, y_test = y[train_index], y[test_index]
        proba = np.repeat(np.mean(y_train), len(y_test))
        losses.append(log_loss(proba, y_test))
    return losses


def cross_validation_log_loss(X, y):
    losses = []
    sss = cross_validation.StratifiedShuffleSplit(y, NUM_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if POINT_ESTIMATE:
            model = linear_model.LogisticRegression()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1].ravel()
            losses.append(log_loss(proba, y_test))
        else:
            data = dict(x=X_train, y=y_train, N=len(y_train),
                        P=X_train.shape[1], x_test=X_test, N_test=len(y_test))
            fit = sm.sampling(data=data, iter=NUM_ITER)
            extr = fit.extract()
            X_train.flags.writeable = False
            results_fn = os.path.join(BASE_DIR, 'cache/crossvalid-{}.pkl'.format(hash(X_train.data.tobytes())))
            import pickle
            pickle.dump(extr, open(results_fn, 'wb'))
            #xbeta_test = np.mean(extr['xbeta_test'], axis=0)
            #np.testing.assert_equal(xbeta_test.shape, y_test.shape)
            #proba = invlogit(xbeta_test)
            #losses.append(log_loss(proba, y_test))
            losses.append(float('nan'))
    return losses


##############################################################################
# Cross-validation
##############################################################################
if __name__ == "__main__":
    print("cross validation")
    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)
    print("y mean:", np.mean(y_train))

    results = collections.defaultdict(list)

    # degenerate model 0
    model_name = "predict average"  # using entire dataset
    print(model_name)
    results[model_name] = cross_validation_log_loss_baseline(y_train)

    # degenerate model 2
    model_name = "article age"
    print(model_name)
    X_tmp = np.atleast_2d(X_train[:, np.in1d(feature_names, ["log_article_age"])])
    results[model_name] = cross_validation_log_loss(X_tmp, y_train)

    # degenerate model 3
    model_name = "article age * bibliographic identifier (VIAF)"
    print(model_name)
    features_include = [
        'log_article_age',
        'category__Wikipedia_articles_with_VIAF_identifiers',
        'category__Wikipedia_articles_with_VIAF_identifiers*log_article_age',
    ]
    X_tmp = np.atleast_2d(X_train[:, np.in1d(feature_names, features_include)])
    results[model_name] = cross_validation_log_loss(X_tmp, y_train)

    # full model
    model_name = "full model"
    print(model_name)
    X_tmp = X_train.copy()
    results[model_name] = cross_validation_log_loss(X_tmp, y_train)

    print(sorted(results.items()))
    results_fn = os.path.join(BASE_DIR, "cache/public-domain-rank-cross-validation-results-n{}-X{}-seed{}.csv".format(len(y_train), 'x'.join(str(dim) for dim in X_train.shape), RANDOM_STATE))
    if DEBUG:
        results_fn = results_fn.replace('.csv', '-DEBUG.csv')
    pd.DataFrame(results).to_csv(results_fn)
