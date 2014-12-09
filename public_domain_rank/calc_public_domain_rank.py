"""
Fit the public domain rank model, after splitting the dataset into
training and test sets.
"""
import hashlib
import itertools
import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import pystan
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing

import public_domain_rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('public_domain_rank')

random.seed(1)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# DEBUG is False by default; one needs tons of memory and time
# If DEBUG is True, subsample aggressively
DEBUG = int(os.environ.get('PUBLIC_DOMAIN_RANK_DEBUG', 0))
SUBSAMPLE_SIZE = int(os.environ.get('PUBLIC_DOMAIN_RANK_SUBSAMPLE_SIZE', 4000))

NUM_ITER = int(os.environ.get('PUBLIC_DOMAIN_RANK_NUM_ITER', 400))
NUM_CHAINS = int(os.environ.get('PUBLIC_DOMAIN_RANK_NUM_CHAINS', 20))
POINT_ESTIMATE = int(os.environ.get('PUBLIC_DOMAIN_RANK_POINT_ESTIMATE', 0))

LANGLINK_MAX_FEATURES = 10
DEATH_DECADE_START = 1910
DEATH_DECADE_STOP = 1950

NUM_RECORDS = 1011304
NUM_RECORDS_DEBUG = 50000
NUM_FEATURES = 239

if DEBUG:
    logger.warn("DEBUG is on")
    dimensions = 'x'.join(str(d) for d in (NUM_RECORDS_DEBUG, NUM_FEATURES))
    dataset_fn = os.path.join(BASE_DIR, 'cache/public-domain-rank-dataset-dim{}.csv'.format(dimensions))
    logging.info("Attempting to load {}".format(dataset_fn))
    try:
        df = pd.read_csv(dataset_fn, index_col=0)
    except OSError:
        logger.info("Did not find cached dataset, rebuilding from scratch")
        df = public_domain_rank.dataset(LANGLINK_MAX_FEATURES, DEATH_DECADE_START, DEATH_DECADE_STOP)
        sample_index = pd.Index(random.sample(list(df.index), NUM_RECORDS_DEBUG))
        df = df.loc[sample_index]
        dataset_fn = os.path.join(BASE_DIR, 'cache/public-domain-rank-dataset-dim{}.csv'.format('x'.join(str(d) for d in df.shape)))
        df.to_csv(dataset_fn)
else:
    dimensions = 'x'.join(str(d) for d in (NUM_RECORDS, NUM_FEATURES))
    dataset_fn = os.path.join(BASE_DIR, 'cache/public-domain-rank-dataset-dim{}.csv'.format(dimensions))
    logging.info("Attempting to load {}".format(dataset_fn))
    try:
        df = pd.read_csv(dataset_fn, index_col=0)
    except OSError:
        logger.info("Did not find cached dataset, rebuilding from scratch")
        df = public_domain_rank.dataset(LANGLINK_MAX_FEATURES, DEATH_DECADE_START, DEATH_DECADE_STOP)
        dataset_fn = os.path.join(BASE_DIR, 'cache/public-domain-rank-dataset-dim{}.csv'.format('x'.join(str(d) for d in df.shape)))
        df.to_csv(dataset_fn)

features_to_drop = ['title', 'redirect_set', 'death_year', 'obp_author', 'obp_digital_editions', 'categories']

# data checks
assert 'topic080' in df.columns
assert 'category__Wikipedia_articles_with_BNF_identifiers' in df.columns
assert 'pagecount' in df.columns
np.testing.assert_(df['topic080'].notnull().values.all())

# make selected transformations
df['log_article_length'] = np.log(df['article_length'])
del df['article_length']
df['log_pagecount'] = np.log(df['pagecount'] + 0.5)
del df['pagecount']
df['log_article_age'] = np.log(df['article_age'] + 0.5)
del df['article_age']
df['log_revisions_per_day'] = np.log(df['revisions_per_day'])
del df['revisions_per_day']

# add interactions
interaction_columns_a = [c for c in df.columns if c not in features_to_drop]
interaction_columns_b = ['log_article_age']
if all(['*' not in c for c in df.columns]):
    for a, b in itertools.product(interaction_columns_a, interaction_columns_b):
        df['{}*{}'.format(a, b)] = df[a] * df[b]
feature_names = [c for c in df.columns if c not in features_to_drop]

TRAIN_START_YEAR = 1910
TRAIN_END_YEAR = 1953  # not inclusive
df_train = df[df.death_year.notnull() & (df.death_year >= TRAIN_START_YEAR) & (df.death_year < TRAIN_END_YEAR)]
print("{} records in the entire set".format(len(df)))
print("{} records in the training set".format(len(df_train)))

# data check
np.testing.assert_(df_train['topic080'].notnull().values.all())

# subsample and balance dataset to speed things up
if DEBUG:
    # include a fixed percentage of positive cases
    positive_index = df_train[df_train.obp_digital_editions > 0].index
    zero_index = df_train[df_train.obp_digital_editions == 0].index
    num_zero = max(SUBSAMPLE_SIZE - len(positive_index), len(positive_index))
    sample_index_zero = pd.Index(random.sample(list(zero_index), num_zero))
    sample_index = pd.Index(positive_index.tolist() + sample_index_zero.tolist())
    np.testing.assert_(sample_index.isin(df_train.index).all())
    if SUBSAMPLE_SIZE < len(sample_index):
        sample_index = pd.Index(random.sample(set(sample_index), SUBSAMPLE_SIZE))
    else:
        sample_index = pd.Index(sample_index)
    np.testing.assert_(sample_index.is_unique)
    np.testing.assert_(sample_index.isin(df_train.index).all())
    df_train = df_train.loc[sample_index]


##############################################################################
# fit model
##############################################################################

scaler = sklearn.preprocessing.StandardScaler()
y_train = (df_train.obp_digital_editions > 0).astype(int).values
X_train = scaler.fit_transform(df_train.drop(features_to_drop, 1).values)

np.testing.assert_(not np.isnan(X_train).any())
np.testing.assert_(not np.isnan(y_train).any())

model_code = """
data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] x;
  int<lower=0, upper=1> y[N];
}
parameters {
  real alpha;
  vector[P] beta;
}
model {
  alpha ~ student_t(7, 0, 5);
  beta ~ student_t(7, 0, 5);
  y ~ bernoulli_logit(alpha + x * beta);
}
"""


##############################################################################
# Fit the public domain rank model and save the raw fit
##############################################################################
if __name__ == "__main__":

    model_fn = os.path.join(BASE_DIR, 'cache/model-{}.pkl'.format(hashlib.md5(model_code.encode('ascii')).hexdigest()))
    try:
        sm = pickle.load(open(model_fn, 'rb'))
    except FileNotFoundError:
        sm = pystan.StanModel(model_code=model_code, model_name='public_domain_rank')
        pickle.dump(sm, open(model_fn, 'wb'))

    # we may save some memory by deleting df
    del df

    data = dict(x=X_train, y=y_train, N=len(y_train), P=X_train.shape[1])

    print("calc public domain rank, X_train shape {}".format(X_train.shape))
    results_hash = hashlib.md5((str(X_train.shape) + model_fn).encode('ascii')).hexdigest()
    results_fn_tmpl = 'cache/public-domain-rank-results-{}-debug{}-mode{}-iter{}-chains{}.pkl'
    results_fn = os.path.join(BASE_DIR, results_fn_tmpl.format(results_hash, int(DEBUG), int(POINT_ESTIMATE), NUM_ITER, NUM_CHAINS))
    extract_fn_tmpl = 'cache/public-domain-rank-results-{}-debug{}-mode{}-iter{}-chains{}-extract.pkl'
    extract_fn = os.path.join(BASE_DIR, extract_fn_tmpl.format(results_hash, int(DEBUG), int(POINT_ESTIMATE), NUM_ITER, NUM_CHAINS))
    if os.path.exists(results_fn):
        raise RuntimeError("Found existing results file, aborting.")
    if POINT_ESTIMATE:
        # FIXME: use pystan here
        model = sklearn.linear_model.LogisticRegression()
        model.fit(X_train, y_train)
        pickle.dump(model, open(results_fn, 'wb'))
    else:
        print('fitting using stan')
        fit = sm.sampling(data=data, iter=NUM_ITER, chains=NUM_CHAINS)
        pickle.dump(fit.extract(), open(extract_fn, 'wb'))
        # FIXME: encountering problems when pickling the fit
        pickle.dump(fit, open(results_fn, 'wb'))
        print(fit)
