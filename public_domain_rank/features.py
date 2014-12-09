"""Defines features used to calculate public domain rank.

Many features are defined relative to others (e.g., averages). The universe of
Wikipedia articles used (defined by pageids) is returned by the function
``pageids_canonical``.

"""
import calendar
import collections
import collections.abc
import logging
import os
import re
import urllib

import bs4  # BeautifulSoup
import mw.lib.title
import numpy as np
import pandas as pd
import sklearn.feature_extraction

import public_domain_rank
import public_domain_rank.topics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# in the 20140402 dump, these pageids are in the categories dump but missing
# from article texts dump, we have to exclude them (5 out of ~1 million)
MISSING_PAGEIDS = [12279576, 31645144, 35118586, 39009411, 42247941]

CATEGORIES_OF_INTEREST = {
    'Wikipedia_articles_with_BNF_identifiers',
    'Wikipedia_articles_with_GND_identifiers',
    'Wikipedia_articles_with_ISNI_identifiers',
    'Wikipedia_articles_with_LCCN_identifiers',
    'Wikipedia_articles_with_VIAF_identifiers',
    'Wikipedia_indefinitely_move-protected_pages',
    'Wikipedia_protected_pages_without_expiry',
    'Good_articles',
    'Featured_articles',
}

DIGITAL_EDITION_BLACKLIST_STRINGS = (
    "page images with commentary at archives.gov",
    "page images at HathiTrust",
    "ultiple formats at Google",
    "ultiple formats at archive.org",
    "uncorrected OCR",
    "PDF at fcla.edu",
)

DIGITAL_EDITION_WHITELIST_STRINGS = (
    "Gutenberg",
)


def pageids_canonical():
    """Return DataFrame with all pageids of interest.

    In particular,

    - No redirects should be in this set.
    - No deleted pages should be in this set.

    These checks exist because Wikipedia is being edited while the database dump
    is executing; minor changes (a redirect, a deletion or two) do happen.

    """
    df = pageids()
    redir = set(redirects().redirect_from)
    df = df[~df.pageid.isin(redir)]
    pageids_with_title = set(titles().index)
    df = df[df.pageid.isin(pageids_with_title)]
    np.testing.assert_(df.set_index('pageid').index.is_unique)

    # these pages are missing from the 20140402 dump
    np.testing.assert_(set(MISSING_PAGEIDS) - set(df.pageid) == set())
    df = df[~df.pageid.isin(set(MISSING_PAGEIDS))]
    return df


def pageids():
    """Return DataFrame with all known pageids (including some redirects)."""
    df = categories()[['pageid']]
    df = df.drop_duplicates().sort('pageid').reset_index(drop=True)
    return df


def redirects():
    redirects_csv = os.path.expanduser(public_domain_rank.config['data']['redirects'])
    # this original table has five columns, we ultimately only want the first two
    # there are three malformed lines which we drop
    df = pd.read_csv(redirects_csv, sep='\t', index_col=False, header=None,
                     error_bad_lines=False, warn_bad_lines=False, usecols=[0, 1, 2])
    df.columns = ['redirect_from', 'namespace', 'redirect_to']
    np.testing.assert_equal(len(df), (df.namespace == 0).sum())
    del df['namespace']
    return df


def redirects_of_interest():
    df = pageids_canonical().set_index('pageid')
    np.testing.assert_(df.index.is_unique)

    n_records = len(df)

    # merge titles
    df_titles = titles()
    df = pd.merge(df, df_titles, left_index=True, right_index=True)
    np.testing.assert_equal(n_records, len(df))

    # merge redirects
    df_redirects = redirects()
    # restrict redirects to universe and add pageid
    df_redirects = df_redirects[df_redirects.redirect_to.isin(df.title)].reset_index(drop=True)
    df_redirects = pd.merge(df_redirects, df_titles, how='left', left_on='redirect_from', right_index=True)
    del df_redirects['redirect_from']  # old redirect_from, was pageid
    df_redirects.rename(columns={'title': 'redirect_from'}, inplace=True)
    df_redirects = df_redirects.set_index('redirect_to')

    # remove NaN; these correspond to pages that have no title in the dump (likely deleted?)
    df_redirects = df_redirects[pd.notnull(df_redirects.redirect_from)]

    # now the df_redirects DataFrame is in a usable form
    df_redirects = df_redirects.groupby(df_redirects.index)['redirect_from'].apply(set)
    df_redirects.name = 'redirect_set'
    return pd.DataFrame(df_redirects)


def categories():
    """Returns DataFrame with pageid and category columns"""
    categories_csv = os.path.expanduser(public_domain_rank.config['data']['categories'])
    df = pd.read_csv(categories_csv,
                     sep='\t',
                     header=None,
                     names=['pageid', 'category'])
    np.testing.assert_equal(df.duplicated().sum(), 0)
    return df


def categories_flat():
    """Returns DataFrame with category sets"""
    df = categories()
    series_grouped = df.groupby('pageid').category.apply(set)
    series_grouped.name = 'categories'
    return pd.DataFrame(series_grouped)


def category_dummies(min_df):
    """Returns DataFrame with category dummy variables.

    Exclude categories associated with fewer than `min_df` articles.

    """
    df = categories()
    pageids_all = df.pageid.unique()
    cat_counter = collections.Counter(df.category.values)
    # categories do not repeat within articles, so this gets us the document freq
    cats_of_interest = {cat for cat, cnt in cat_counter.items() if cnt >= min_df}

    # drop categories that we are not interested in
    df = df[df.category.isin(cats_of_interest)]

    # group and use one-hot encoding
    series_grouped = df.groupby('pageid').category.apply(lambda lst: tuple((k, 1) for k in lst))
    del df
    # NOTE: sparse matrices need special handling with pandas so avoid them here
    v = sklearn.feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform([dict(tuples) for tuples in series_grouped])
    pageids = series_grouped.index
    del series_grouped
    category_names = ['category__{}'.format(name) for name in v.get_feature_names()]
    df = pd.DataFrame(X, columns=category_names, index=pageids)

    # if there are any pageids that have no categories associated with them (and
    # having appropriate `min_df`), add rows of zeros for them.
    if len(set(pageids_all) - set(pageids)) > 0:
        pageids_missing = list(set(pageids_all) - set(pageids))
        n_missing = len(pageids_missing)
        msg = "category_dummies: no categories for {} pages (of {})".format(n_missing, len(pageids_all))
        logger.warn(msg)
        X_missing = np.zeros((len(pageids_missing), X.shape[1]))
        df_missing = pd.DataFrame(X_missing, columns=category_names, index=pageids_missing)
        df = pd.concat([df, df_missing])

    return df


def category_dummies_of_interest():
    """Returns DataFrame with category dummy variables.

    Exclude categories associated with fewer than `min_df` articles.

    """
    df = categories()
    pageids_all = df.pageid.unique()

    # drop categories that we are not interested in
    df = df[df.category.isin(CATEGORIES_OF_INTEREST)]

    # group and use one-hot encoding
    series_grouped = df.groupby('pageid').category.apply(lambda lst: tuple((k, 1) for k in lst))
    del df

    # NOTE: sparse matrices need special handling with pandas so avoid them here
    v = sklearn.feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform([dict(tuples) for tuples in series_grouped])
    pageids = series_grouped.index
    del series_grouped
    category_names = ['category__{}'.format(name) for name in v.get_feature_names()]
    df = pd.DataFrame(X, columns=category_names, index=pageids)

    # if there are any pageids that have no categories associated with them (and
    # having appropriate `min_df`), add rows of zeros for them.
    if len(set(pageids_all) - set(pageids)) > 0:
        pageids_missing = list(set(pageids_all) - set(pageids))
        n_missing = len(pageids_missing)
        msg = "category_dummies: no categories for {} pages (of {})".format(n_missing, len(pageids_all))
        logger.warn(msg)
        X_missing = np.zeros((len(pageids_missing), X.shape[1]))
        df_missing = pd.DataFrame(X_missing, columns=category_names, index=pageids_missing)
        df = pd.concat([df, df_missing])

    return df


def titles():
    """Returns DataFrame with all page titles indexed by pageid.

    Included in these titles are titles that redirect to different pages."""
    titles_fn = os.path.expanduser(public_domain_rank.config['data']['titles'])
    df = pd.read_csv(titles_fn, sep='\t', header=None, index_col=0, usecols=[0, 1],
                     error_bad_lines=False, warn_bad_lines=False)
    df.index.name = 'pageid'
    df.columns = ['title']
    np.testing.assert_(df.index.is_unique)
    return df


def topics():
    """Returns DataFrame with topic distributions (indexed by pageid)"""
    df = public_domain_rank.topics.theta()
    np.testing.assert_(df.index.is_unique)
    return df


def imagelink_counts():
    """Returns DataFrame with imagelink counts (indexed by pageid)"""
    imagelinks_fn = os.path.expanduser(public_domain_rank.config['data']['imagelinks'])
    df = pd.read_csv(imagelinks_fn, sep='\t', header=None, index_col=0, usecols=[0, 1],
                     error_bad_lines=False, warn_bad_lines=False)
    np.testing.assert_(df.index.is_unique)
    df.index.name = 'pageid'
    df.columns = ['imagelink_count']
    return df


def langlinks():
    """Returns DataFrame with language links (indexed by pageid)"""
    langlinks_fn = os.path.expanduser(public_domain_rank.config['data']['langlinks'])
    df = pd.read_csv(langlinks_fn, sep='\t', header=None, usecols=[0, 1],
                     error_bad_lines=False, warn_bad_lines=False)
    df.columns = ['pageid', 'langlink']
    return df


def langlink_counts():
    """Returns DataFrame with lanuagelink counts (indexed by pageid)"""
    langlinks_fn = os.path.expanduser(public_domain_rank.config['data']['langlinks'])
    df = pd.read_csv(langlinks_fn, sep='\t', header=None, usecols=[0, 1],
                     error_bad_lines=False, warn_bad_lines=False)
    df.columns = ['pageid', 'langlink']
    df = pd.DataFrame(df.groupby('pageid').size())
    df.columns = ['langlink_count']
    return df


def langlink_dummies(max_features):
    """Returns DataFrame with language link dummy variables.

    Exclude languages associated with fewer than `min_df` articles.

    """
    df = langlinks()
    pageids_all = df.pageid.unique()
    langlink_counter = collections.Counter(df.langlink.values)
    # langlinks do not repeat within articles, so this gets us the document freq
    langlinks_of_interest = {langlink for langlink, cnt in langlink_counter.most_common(max_features)}

    # drop langlinks that we are not interested in
    df = df[df.langlink.isin(langlinks_of_interest)]

    # group and use one-hot encoding
    series_grouped = df.groupby('pageid').langlink.apply(lambda lst: tuple((k, 1) for k in lst))
    del df
    # NOTE: sparse matrices need special handling with pandas so avoid them here
    v = sklearn.feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform([dict(tuples) for tuples in series_grouped])
    pageids = series_grouped.index
    del series_grouped
    langlink_names = ['langlink__{}'.format(name) for name in v.get_feature_names()]
    df = pd.DataFrame(X, columns=langlink_names, index=pageids)

    # if there are any pageids that have no langlinks associated with them (and
    # having appropriate `min_df`), add rows of zeros for them.
    if len(set(pageids_all) - set(pageids)) > 0:
        pageids_missing = list(set(pageids_all) - set(pageids))
        n_missing = len(pageids_missing)
        msg = "langlink_dummies: no langlinks for {} pages (of {})".format(n_missing, len(pageids))
        logger.warn(msg)
        X_missing = np.zeros((len(pageids_missing), X.shape[1]))
        df_missing = pd.DataFrame(X_missing, columns=langlink_names, index=pageids_missing)
        df = pd.concat([df, df_missing])
    return df


_VIAF_RE = re.compile(r'(?:viaf|VIAF) ?= ?([0-9]+)')


def _extract_viaf(text):
    """Extract VIAF from article full text

    Returns
    -------
    viaf : {int, nan}
      nan if not found
    """
    if 'viaf' in text or 'VIAF' in text:
        try:
            viaf = int(_VIAF_RE.findall(text)[0])
        except IndexError:
            # Found "VIAF" in text but no number; this happens when there's
            # some bibliographic data but not VIAF
            viaf = float('nan')
    else:
        viaf = float('nan')
    return viaf


def viafs(pageids):
    """Return DataFrame with VIAFs found in articles identified by `pageids`.
    DataFrame is indexed by pageid
    'nan' is used if no VIAF could be found in article text
    """
    pageids = set(pageids)
    texts_dir = os.path.expanduser(public_domain_rank.config['data']['texts-dir'])
    pageid_from_fn = lambda fn: int(os.path.basename(fn).replace('.txt', ''))
    text_fns = [os.path.join(texts_dir, fn) for fn in os.listdir(texts_dir) if pageid_from_fn(fn) in pageids]
    if len(pageids) != len(text_fns):
        raise ValueError("Failed to find text for all pageids.")
    text_pageids = [pageid_from_fn(fn) for fn in text_fns]
    text_viafs = [_extract_viaf(open(fn).read()) for fn in text_fns]
    return pd.DataFrame(dict(pageid=text_pageids, viaf=text_viafs)).set_index('pageid')


def article_lengths(pageids=None):
    """Return DataFrame with article lengths for articles identified by `pageids`.

    DataFrame is indexed by pageid. An error is raised if no article is found.
    """
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['pageid-title-timestamp-length'])
    df = pd.read_csv(csv_fn, sep='\t').set_index('pageid')[['article_length']]
    if pageids is not None:
        pageids = set(pageids)
        df = df[df.index.isin(pageids)]
        if len(df) < len(pageids):
            raise ValueError("Failed to find article lengths for all pageids.")
    return df


def last_revision(pageids=None):
    """Return DataFrame with last revision timestamp in seconds from the epoch"""
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['pageid-title-timestamp-length'])
    df = pd.read_csv(csv_fn, sep='\t').set_index('pageid')[['revision_timestamp']]
    df['revision_timestamp'] = pd.to_datetime(df['revision_timestamp'], format="%Y%m%d%H%M%S").apply(lambda dt: calendar.timegm(dt.utctimetuple()))
    if pageids is not None:
        pageids = set(pageids)
        df = df[df.index.isin(pageids)]
        if len(df) < len(pageids):
            raise ValueError("Failed to find text for all pageids.")
    return df


def article_age(pageids=None):
    """Return DataFrame with days since first revision"""
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['revision-metadata'])
    df = pd.read_csv(csv_fn, sep='\t').set_index('pageid')[['first_revision']]
    if pageids is not None:
        pageids = set(pageids)
        df = df[df.index.isin(pageids)]
        if len(df) < len(pageids):
            raise ValueError("Failed to find article age for all pageids.")
    df['article_age'] = (df.first_revision.max() - df.first_revision) / (60 * 60 * 24)
    del df['first_revision']
    return df


def revisions_per_day(pageids=None, cutoff_days=30):
    """Return DataFrame with average revisions per day for articles with `pageids`.

    Articles that have existed for fewer than `cutoff_days` have at maximum of
    1 revision per day. This is to prevent cases where an article was created,
    for example, two hours before the database dump and would otherwise be
    recorded as having an average of 12 revisions per day.

    NOTE: for purposes of calculation, "today" is one second after the most
    recent revision among all the articles for which revision data is collected.
    For Wikipedia this is a reasonable assumption and it allows the script to
    work unmodified with different dumps.
    """
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['revision-metadata'])
    df = pd.read_csv(csv_fn, sep='\t').set_index('pageid')
    # in seconds since the epoch, add one to avoid a division by zero error for that page
    today_timestamp = max(df.first_revision) + 1
    if pageids is not None:
        pageids = set(pageids)
        df = df[df.index.isin(pageids)]
        if len(df) < len(pageids):
            raise ValueError("Failed to find text for all pageids.")
    revisions_per_day = []
    for pageid, (n_revisions, first_revision) in df.iterrows():
        days_since_creation = (today_timestamp - first_revision) / (3600 * 24)
        if days_since_creation < cutoff_days:
            revisions_per_day.append(min(1, n_revisions / days_since_creation))
        else:
            revisions_per_day.append(n_revisions / days_since_creation)
    return pd.DataFrame(dict(revisions_per_day=revisions_per_day), index=df.index)


def _onlinebookspage_authors_with_wikipedia_link():
    """Returns data indicating the presence of one or more digital editions
    of works by an author on The Online Books Page (OBP).

    This particular call only identifies authors who already have a Wikipedia
    page indicated in the OBP.
    """
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['onlinebookspage'])
    df_obp = pd.read_csv(csv_fn)[['author', 'link', 'wikipedia_link', 'num_titles', 'page']]
    df_obp = df_obp[df_obp.wikipedia_link.notnull()]
    obp_wp_link_prefix = 'http://en.wikipedia.org/wiki/'
    df_obp['wikipedia_title'] = [s.replace(obp_wp_link_prefix, '') for s in df_obp.wikipedia_link]
    del df_obp['wikipedia_link']
    df_obp['wikipedia_title'] = [mw.lib.title.normalize(urllib.parse.unquote(t)) for t in df_obp.wikipedia_title]

    # aggressively drop duplicate entries from OBP, there are just a couple
    # for example, Xuanzang has two entries with same Wikipedia link
    df_obp = df_obp[~df_obp.wikipedia_title.duplicated()]
    assert all(~df_obp.wikipedia_title.duplicated())

    df = pageids_canonical().set_index('pageid').join(titles(), how="left")
    assert all(df.title.notnull())
    assert df.index.is_unique

    # first remove all those that match perfectly, then consider additional methods of matching
    df_matched = pd.merge(df.reset_index(), df_obp, left_on='title', right_on='wikipedia_title').set_index('pageid')
    assert df_matched.index.is_unique

    # NOTE: 2014-08-21 attempted to see if OBP wikipedia links might be linking
    # to old titles (i.e., redirects) and found that there were zero such links
    df_matched = df_matched[['author', 'num_titles', 'page']]
    df_matched.columns = ['obp_author', 'obp_n_titles', 'obp_html']
    return df_matched


def _onlinebookspage_authors_without_wikipedia_link():
    """Returns data indicating the presence of one or more digital editions
    of works by an author on The Online Books Page (OBP).

    This particular call only identifies authors whose OBP page does not link
    to a Wikipedia page. These are typically manual matches.
    """
    csv_fn = os.path.expanduser(public_domain_rank.config['data']['onlinebookspage'])
    df_obp = pd.read_csv(csv_fn)[['author', 'n_titles', 'page']]
    df_obp.columns = ['obp_author', 'obp_n_titles', 'obp_html']

    csv_fn = os.path.expanduser(public_domain_rank.config['data']['onlinebookspage-wikipedia-mapping'])
    df = pd.read_csv(csv_fn)
    df = df[df.pageid.notnull()]
    df = pd.merge(df, df_obp, on='obp_author', how='left')
    # aggressively drop duplicate entries as they reflect rare duplicates in OBP
    df = df[~df.pageid.duplicated()]
    df = df.set_index('pageid')
    assert df.index.is_unique
    assert all(~df.obp_author.duplicated())
    df = df[['obp_author', 'obp_n_titles', 'obp_html']]
    return df


def onlinebookspage():
    """Returns data indicating the presence of one or more digital editions
    of works by an author on The Online Books Page (OBP).
    """
    df_obp_with_wp = _onlinebookspage_authors_with_wikipedia_link()
    assert df_obp_with_wp.index.is_unique
    assert all(~df_obp_with_wp.obp_author.duplicated())

    df_obp_without_wp = _onlinebookspage_authors_without_wikipedia_link()
    assert df_obp_without_wp.index.is_unique
    assert all(~df_obp_without_wp.obp_author.duplicated())

    assert all(df_obp_with_wp.columns == df_obp_without_wp.columns)

    # discard any overlap
    df_obp_without_wp = df_obp_without_wp.loc[df_obp_without_wp.index - df_obp_with_wp.index]

    df = pd.concat([df_obp_with_wp, df_obp_without_wp])
    assert df.index.is_unique
    assert all(~df.obp_author.duplicated())

    return df


def digital_editions():
    """Returns data indicating the presence of one or more digital editions of
    works by an author on The Online Books Page (OBP). A digital edition is, in
    essence, corrected OCR.
    """
    df = onlinebookspage()

    def _works_from_html(html):
        return [li.get_text() for li in bs4.BeautifulSoup(html).findAll('li')]

    def _digital_editions(html):
        works = _works_from_html(html)
        filtered = []
        for work in works:
            if any(s in work for s in DIGITAL_EDITION_WHITELIST_STRINGS):
                filtered.append(work)
            elif not any(s in work for s in DIGITAL_EDITION_BLACKLIST_STRINGS):
                filtered.append(work)
        return filtered

    def digital_edition_count(html):
        return len(_digital_editions(html))

    df['obp_digital_editions'] = df.obp_html.apply(digital_edition_count).astype(int)

    # drop obp_n_titles
    return df[['obp_author', 'obp_digital_editions']]


def death_year():
    """Returns DataFrame with death year indexed by pageid"""
    df = categories()
    deaths_cat_re = re.compile(r'^[0-9]{1,4}s?[_ ][Dd]eaths$')
    df = df[[deaths_cat_re.match(c) is not None for c in df.category]]
    df = df.groupby('pageid').first()
    death_year_re = re.compile(r'[0-9]+')
    death_year = df.category.apply(lambda c: int(death_year_re.search(c).group()))
    df['death_year'] = death_year
    assert all(df.death_year.notnull())
    return df[['death_year']]


def death_decade_dummies(min_year, max_year):
    """Returns DataFrame with death year decade indicator variables

    If a pageid has no death year associated with it, it is assumed to be
    greater than `max_year`.
    """
    df = death_year()
    bins = np.arange(min_year, max_year + 1, 10).astype(int)
    bin_labels = ['death_year_lt_{:d}'.format(min(bins))]
    bin_labels += ['death_year_{:d}s'.format(n) for n in bins[:-1]]
    bin_labels += ['death_year_gte_{:d}'.format(max(bins))]
    bin_labels = np.array(bin_labels)
    df['death_decade'] = bin_labels[np.digitize(df.death_year, bins)]
    df_pageid = pageids_canonical().set_index('pageid')
    df_final = df_pageid.join(df[['death_decade']])
    label_nan_fill = bin_labels[-1]
    df_final = df_final.fillna(label_nan_fill)
    assert all(df_final.notnull())
    return pd.get_dummies(df_final.death_decade)


def pagecounts():
    """Returns DataFrame with pagecounts"""
    pagecounts_fn = os.path.expanduser(public_domain_rank.config['data']['pagecounts_of_interest'])
    df = pd.read_csv(pagecounts_fn, index_col=0)
    return df[['pagecount']]
