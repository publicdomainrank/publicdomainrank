# Public Domain Rank Data

The following datasets are are committed to the code repository.

- ``data/onlinebookspage-wikipedia-mapping.csv`` contains additional mappings
  between Online Book Page entries and Wikipedia pages. These are collected
  manually for the most part. Most matching is done automatically given The
  Online Books Page's own data (it often provides a Wikipedia link on authors'
  pages).

# Public Domain Rank External Data

The code uses several datasets that are too large to be committed to the code
repository. The location of these datasets needs to be specified in ``config.ini``.

The datasets and md5 hashes are the following:

    13b47b6090d9989d7254988d07ab46cb  enwiki-20140402-ns0-redirects.csv
    47e2292e2b58f5c2edd74ad2a507711f  enwiki-20140402-pageid-title-timestamp-length.csv
    cac196d4782bace7c94facf53cd0a07f  enwiki-20140402-categorylinks-pages-persons.csv
    5f798b43492e667ed18f467c976d20d4  enwiki-20140402-revision-metadata.csv
    b2722d5223bff85151cb441865328376  enwiki-20140402-imagelinks.csv
    b7a109f2eb11b6b5af1ada78e1dd011d  enwiki-20140402-langlinks.csv
    c95fb431e9c8088920651fbd479cca24  onlinebookspage-20140516-authors.csv
    c8929669e1e07ee1b9506663d87024fe  enwiki-20140402-pagecounts-of-interest.csv
    8cade3083b5d100e407e431a99bdb852  pagecounts-201209-to-201308-60-days-sorted-groupby-title.h5
    55c2dcb433494a688a28bf64f3c97a4a  enwiki-20140402-pages-articles.xml.bz2
    3134c33165729a23adab922179aadc6b  enwiki-20140402-ns0-titles.csv
    ae4ab3571eff34565119e2909c9fc879  enwiki-20140402-pages-articles-texts-stemmed-mindf200.documents
    c17fcb031c021ebd3171cf8ed8d1af68  enwiki-20140402-pages-articles-texts-stemmed-mindf200.ldac
    dcae53e4542ee19410fe1c83202f3303  enwiki-20140402-pages-articles-texts-stemmed-mindf200.tokens
    6d4d061140c91951a7e93acbb273983d  enwiki-20140402-pages-articles-texts-stemmed-mindf200-fit-K200.ndt
    ae960928aed8fbd6d51e13679006ebd8  enwiki-20140402-pages-articles-texts-stemmed-mindf200-fit-K200.nwt
    8b9d20ca5035171fc40e09b9632a8d50  enwiki-20140402-stub-meta-history.xml.gz

To create most of these datasets you will need access to the 2014-4-2 Wikipedia
database "dump". For the remainder, the data is being distributed as two separate torrents:

- `data/pagecounts.torrent`
- `data/public-domain-rank.torrent`

Descriptions of these files follow.

Online Books Page
-----------------

The file ``onlinebookspage-...-authors.csv`` is constructed by converting the
``.json`` dumps from the scrapy spider found in ``onlinebooks-spider`` as in

    import json
    records = json.load(open('onlinebooks-spider/crawls/authors.json'))
    df = pd.DataFrame.from_records(records)
    df.to_csv('onlinebookspage-authors.csv', index=False)

The particular scrape was run on 2014-5-16.

Article texts and titles
------------------------

The article texts and the file ``enwiki-20140402-pageid-title-timestamp-length.csv``
may be extracted by running the command: ``python -m public_domain_rank.extract_text``.

Article texts and topic model
-----------------------------

The full text of articles has been collected in the following files:

    ae4ab3571eff34565119e2909c9fc879  enwiki-20140402-pages-articles-texts-stemmed-mindf200.documents
    c17fcb031c021ebd3171cf8ed8d1af68  enwiki-20140402-pages-articles-texts-stemmed-mindf200.ldac
    dcae53e4542ee19410fe1c83202f3303  enwiki-20140402-pages-articles-texts-stemmed-mindf200.tokens

Words occurring in fewer than 200 documents (roughly 0.1% of the corpus) have
been excluded from the vocabulary, as have words containing numbers or
underscores. Words have been stemmed using the
``nltk.stem.snowball.EnglishStemmer`` and tokenized using the pattern
``(?u)\b\w\w+\b``.

A topic model of the corpus has been fit using
[hca](http://www.mloss.org/software/view/527/) version 0.5. ``hca`` was invoked
as follows:

```
~/bin/hca -v -q16 -K100 -C5000 -s5 -N1000000,10000000 -c100 -fldac ~/work/hca/data/enwiki-20140402-pages-articles-texts-stemmed-mindf200 ~/work/hca/fits/fit-enwiki-20140402-pages-articles-texts-stemmed-mindf200-K100-20140709
```

The model was run until convergence (assessed visually by monitoring the
complete log likelihood). This yields the following files:

<!-- TODO -->
<!-- TODO: ACTUALLY CHECK LOGLIKELIHOOD -->


Revisions
---------
Counting the revisions associated with a pageid requires sweeping through
``enwiki-20140402-stub-meta-history.xml.gz``, which is 39G. The script
``extract_revisions.py`` will extract basic summary statistics for the pages of
interest. This script should take less than 48 hours to run and generates a file
``revision-metadata.csv``.

Pagecounts
----------

Counting the page views an article and all its aliases (redirects) receive takes
quite some time so we prefetch these. The file is generated by running the
following:

    python -m  public_domain_rank.extract_pagecounts_of_interest enwiki-20140402-pagecounts-of-interest.csv

Pagecounts depend on the following 26G file, which contains aggregated
pagecounts data for 60 random days between September 1, 2012 and August 31,
2013:

    8cade3083b5d100e407e431a99bdb852  pagecounts-201209-to-201308-60-days-sorted-groupby-title.h5

Database queries
----------------

Several datasets are constructed from queries to a MariaDB database that has
been loaded with the relevant database dumps from Wikipedia.
For example, the table ``categorylinks`` is available in the file
``enwiki-20140402-categorylinks.sql.gz`` which may be loaded into a MariaDB
database with the command ``zcat enwiki-20140402-categorylinks.sql.gz | mysql
enwiki``. For the larger of these dumps this is a very time consuming process.

The sections below describe the specific queries.

### Pages and categories

The following MariaDB commands reconstruct the essential dataset for the
project: the list of articles of individuals with a birth year or death who died
after 1000 CE with their associated categories. We also include anyone with
a bibliographic identifier, in the event that birth and death years are unknown.


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Retrieve all categories for pages with one or more of the following
    % bibliographic identifiers:
    % - Wikipedia_articles_with_BNF_identifiers
    % - Wikipedia_articles_with_GND_identifiers
    % - Wikipedia_articles_with_ISNI_identifiers
    % - Wikipedia_articles_with_LCCN_identifiers
    % - Wikipedia_articles_with_NLA_identifiers
    % - Wikipedia_articles_with_SELIBR_identifiers
    % - Wikipedia_articles_with_ULAN_identifiers
    % - Wikipedia_articles_with_VIAF_identifiers"
    % - ????_births
    % - ????_deaths
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the categorylinks table in enwiki has columns cl_from and cl_to
    % For example, George Orwell's pageid is 11891. If we query for his pageid
    % using the following query we get the (partial) results below.
    %
    % SELECT cl_from, cl_to FROM categorylinks WHERE cl_from = 11891;
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % +---------+--------------------------------------------------------------+
    % | cl_from | cl_to                                                        |
    % +---------+--------------------------------------------------------------+
    % |   11891 | 1903_births                                                  |
    % |   11891 | 1950_deaths                                                  |
    % |   11891 | Wikipedia_articles_with_BNF_identifiers                      |
    % |   11891 | Wikipedia_articles_with_GND_identifiers                      |
    % |   11891 | Wikipedia_articles_with_ISNI_identifiers                     |
    % |   11891 | Wikipedia_articles_with_LCCN_identifiers                     |
    % |   11891 | Wikipedia_articles_with_NLA_identifiers                      |
    % |   11891 | Wikipedia_articles_with_SELIBR_identifiers                   |
    % |   11891 | Wikipedia_articles_with_ULAN_identifiers                     |
    % |   11891 | Wikipedia_articles_with_VIAF_identifiers                     |
    % |   11891 | Wikipedia_indefinitely_move-protected_pages                  |
    % +---------+--------------------------------------------------------------+

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % create a temporary view with page ids that have bibliographic identifiers
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    CREATE OR REPLACE VIEW pageids_of_interest AS
    SELECT DISTINCT
      cl_from
    FROM categorylinks
    WHERE
      cl_to LIKE "____\_births" OR
      cl_to LIKE "____\_deaths" OR
      cl_to IN ("Wikipedia_articles_with_BNF_identifiers",
              "Wikipedia_articles_with_GND_identifiers",
              "Wikipedia_articles_with_ISNI_identifiers",
              "Wikipedia_articles_with_LCCN_identifiers",
              "Wikipedia_articles_with_NLA_identifiers",
              "Wikipedia_articles_with_SELIBR_identifiers",
              "Wikipedia_articles_with_ULAN_identifiers",
              "Wikipedia_articles_with_VIAF_identifiers") ORDER BY cl_from;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % collapse categories for all pages
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    SELECT DISTINCT
      cl_from, categorylinks.cl_to
    FROM
      pageids_of_interest
    JOIN
      categorylinks
    USING (cl_from)
    ORDER BY
      cl_from, categorylinks.cl_to
    INTO OUTFILE "/tmp/enwiki-20140402-categorylinks-pages-persons.csv";

### Redirects

The file ``enwiki-20140402-ns0-redirects.csv`` is created with the following query:

    SELECT * FROM redirect WHERE rd_namespace = 0 INTO OUTFILE "/tmp/enwiki-20140402-ns0-redirects.csv";

### Titles

The file ``enwiki-20140402-ns0-titles.csv`` is created with the following query:

    SELECT page_id, page_title FROM page WHERE page_namespace = 0 INTO OUTFILE "/tmp/enwiki-20140402-ns0-titles.csv";

### Image links

The file ``enwiki-20140402-imagelinks.csv`` is created with the following query:

    SELECT DISTINCT il_from, count(*) as n_links FROM imagelinks GROUP BY il_from INTO OUTFILE "/tmp/enwiki-20140402-imagelinks.csv

### Language links

The file ``enwiki-20140402-langlinks.csv`` is created with the following query:

    SELECT DISTINCT ll_from, ll_lang FROM langlinks INTO OUTFILE "/tmp/enwiki-20140402-langlinks.csv";