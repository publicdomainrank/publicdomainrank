import os
import unittest

import pandas as pd
import numpy as np

import public_domain_rank.features

PAGEID_ORWELL = 11891
PAGEID_WOOLF = 32742
PAGEID_DOEBLIN = 896457
PAGEID_EINSTEIN = 736
PAGEID_OCONNOR = 96429
PAGEID_WRIGHT = 43153  # Richard Wright
PAGEID_LENIN = 11015252
PAGEID_TROTSKY = 17888
PAGEID_LADY_MORGAN = 1098693


class TestTitles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.titles()

    def test_flannery_oconnor(self):
        df = self.df
        title = "Flannery_O'Connor"
        record = df[df.title == title]
        self.assertEqual(record.title.ravel()[0], title)
        self.assertEqual(int(record.index), PAGEID_OCONNOR)


class TestRedirects(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_raw = public_domain_rank.features.redirects()

        df = public_domain_rank.features.redirects_of_interest()
        df['redirect_set'] = df['redirect_set'].fillna(set())
        df['n_redirect'] = df['redirect_set'].apply(len)
        cls.df = df

    def test_n_redirects(self):
        # Wikipedia should have plenty of redirects
        df = self.df
        self.assertGreater(sum(df.n_redirect), 300000)

    def test_virginia_woolf(self):
        # "Virginia woolf" (note lowercase "w") having pageid 512249 redirects to "Virginia Woolf"
        df = self.df_raw
        pageid = 512249
        record = df[df.redirect_from == pageid]
        title = "Virginia_Woolf"
        self.assertEqual(record.redirect_to.ravel()[0], title)

    def test_orwell_woolf(self):
        df = self.df
        orwell_n_redirect = df.loc['George_Orwell'].n_redirect
        woolf_n_redirect = df.loc['Virginia_Woolf'].n_redirect
        self.assertGreater(orwell_n_redirect, 5)
        self.assertGreater(woolf_n_redirect, 5)
        self.assertLess(orwell_n_redirect, woolf_n_redirect)


class TestCategoriesFlat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.categories_flat()

    def test_categories_flat(self):
        df = self.df
        self.assertIsInstance(df.loc[PAGEID_EINSTEIN].categories, set)
        self.assertGreater(len(df.loc[PAGEID_EINSTEIN].categories),
                           len(df.loc[PAGEID_DOEBLIN].categories))


class TestCategories(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.categories()

    def test_alfred_doeblin(self):
        df = self.df
        pageid = PAGEID_DOEBLIN
        page_categories = set(df[df.pageid == pageid].category)
        self.assertGreater(len(page_categories), 20)
        categories_of_interest = ["Converts to Roman Catholicism from Judaism",
                                  "19th-century German people",
                                  "Exilliteratur writers"]
        for cat in categories_of_interest:
            self.assertIn(cat.replace(' ', '_'), page_categories)

    def test_richard_wright(self):
        df = self.df
        pageid = PAGEID_WRIGHT
        page_categories = set(df[df.pageid == pageid].category)
        self.assertGreater(len(page_categories), 20)
        categories_of_interest = ["Spingarn Medal winners",
                                  "Writers from Mississippi",
                                  "American socialists"]
        for cat in categories_of_interest:
            self.assertIn(cat.replace(' ', '_'), page_categories)

    def test_flannery_oconnor(self):
        df = self.df
        pageid = PAGEID_OCONNOR
        page_categories = set(df[df.pageid == pageid].category)
        self.assertGreater(len(page_categories), 20)
        categories_of_interest = ["Christian novelists",
                                  "Women short story writers",
                                  "National Book Award winners",
                                  "Burials at Memory Hill Cemetery"]
        for cat in categories_of_interest:
            self.assertIn(cat.replace(' ', '_'), page_categories)


class TestCategoryDummies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1957_deaths occurs n times (used in test below)
        cls.df = public_domain_rank.features.category_dummies(min_df=2300)

    def test_category_dummies_merge(self):
        # categories dummies are exhaustive because categories were used to define
        # the universe of pageids.
        df_category_dummies = self.df
        df = public_domain_rank.features.pageids_canonical().set_index('pageid')
        np.testing.assert_(df.index.is_unique)
        np.testing.assert_(df_category_dummies.index.is_unique)
        self.assertTrue(df.notnull().values.all())
        df_merged = pd.merge(df, df_category_dummies, left_index=True, right_index=True, how='left')
        np.testing.assert_equal(len(df), len(df_merged))
        self.assertTrue(df_merged.notnull().values.all())

    def test_alfred_doeblin(self):
        df = self.df
        series = df.loc[PAGEID_DOEBLIN]
        category_columns = series[series > 0].index
        categories = {c.replace('category__', '') for c in category_columns}
        self.assertIn('1957_deaths', categories)
        self.assertIn('Wikipedia_articles_with_VIAF_identifiers', categories)

    def test_richard_wright(self):
        df = self.df
        series = df.loc[PAGEID_WRIGHT]
        category_columns = series[series > 0].index
        categories = {c.replace('category__', '') for c in category_columns}
        self.assertIn('20th-century_American_novelists', categories)
        self.assertIn('Wikipedia_articles_with_VIAF_identifiers', categories)

    def test_flannery_oconnor(self):
        df = self.df
        series = df.loc[PAGEID_OCONNOR]
        category_columns = series[series > 0].index
        categories = {c.replace('category__', '') for c in category_columns}
        self.assertIn('20th-century_American_novelists', categories)
        self.assertIn('Wikipedia_articles_with_VIAF_identifiers', categories)


class TestImagelinkCounts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.imagelink_counts()

    def test_comparisons(self):
        df = self.df
        pageid_doeblin = PAGEID_DOEBLIN
        pageid_einstein = PAGEID_EINSTEIN
        pageid_oconnor = PAGEID_OCONNOR
        self.assertGreater(df.loc[pageid_einstein].imagelink_count,
                           df.loc[pageid_doeblin].imagelink_count)
        self.assertGreater(df.loc[pageid_einstein].imagelink_count,
                           df.loc[pageid_oconnor].imagelink_count)


class TestLanglinkCounts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.langlink_counts()

    def test_comparisons(self):
        df = self.df
        pageid_doeblin = PAGEID_DOEBLIN
        pageid_einstein = PAGEID_EINSTEIN
        pageid_oconnor = PAGEID_OCONNOR
        self.assertGreater(df.loc[pageid_einstein].langlink_count, 0)
        self.assertGreater(df.loc[pageid_oconnor].langlink_count, 0)
        self.assertGreater(df.loc[pageid_doeblin].langlink_count, 0)
        self.assertGreater(df.loc[pageid_einstein].langlink_count,
                           df.loc[pageid_doeblin].langlink_count)
        self.assertGreater(df.loc[pageid_einstein].langlink_count,
                           df.loc[pageid_oconnor].langlink_count)

    def test_langlink_counts_merge(self):
        df_langlink_counts = self.df
        df = public_domain_rank.features.pageids_canonical().set_index('pageid')
        df_merged = pd.merge(df, df_langlink_counts, left_index=True, right_index=True, how='left')
        # we should have NaNs since many pages lack translations
        np.testing.assert_equal(len(df), len(df_merged))
        self.assertTrue(not df_merged.notnull().values.all())


class TestLanglinkDummies(unittest.TestCase):

    common_langs = ['fr', 'de', 'it', 'es', 'pl', 'ru']

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.langlink_dummies(max_features=50)

    def test_langlink_dummies_merge(self):
        df_langlink_dummies = self.df
        df = public_domain_rank.features.pageids_canonical().set_index('pageid')
        np.testing.assert_(df.index.is_unique)
        np.testing.assert_(df_langlink_dummies.index.is_unique)
        self.assertTrue(df.notnull().values.all())
        df_merged = pd.merge(df, df_langlink_dummies, left_index=True, right_index=True, how='left')
        self.assertTrue(not df_merged.notnull().values.all())
        df_merged = df_merged.fillna(0)
        self.assertTrue(df_merged.notnull().values.all())
        np.testing.assert_equal(len(df), len(df_merged))

        # for thoroughness, consider pages having no translations
        self.assertEqual(df_merged.loc[42365113].values.sum(), 0)
        self.assertEqual(df_merged.loc[7797].values.sum(), 0)

    def test_alfred_doeblin(self):
        df = self.df
        common_langs = self.common_langs
        pageid = PAGEID_DOEBLIN
        series = df.loc[pageid]
        langlink_columns = series[series > 0].index
        langlinks = {c.replace('langlink__', '') for c in langlink_columns}
        for lang in common_langs:
            self.assertIn(lang, langlinks)

    def test_einstein(self):
        df = self.df
        common_langs = self.common_langs
        pageid = PAGEID_EINSTEIN
        series = df.loc[pageid]
        langlink_columns = series[series > 0].index
        langlinks = {c.replace('langlink__', '') for c in langlink_columns}
        for lang in common_langs:
            self.assertIn(lang, langlinks)

    def test_oconnor(self):
        df = self.df
        common_langs = self.common_langs
        pageid = PAGEID_OCONNOR
        series = df.loc[pageid]
        langlink_columns = series[series > 0].index
        langlinks = {c.replace('langlink__', '') for c in langlink_columns}
        for lang in common_langs:
            self.assertIn(lang, langlinks)


class TestVIAF(unittest.TestCase):

    def test_viaf(self):
        pageids = [PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR]
        df = public_domain_rank.features.viafs(pageids)
        # these are the VIAF numbers corresponding to the pageids above
        self.assertEqual(df.loc[PAGEID_EINSTEIN].viaf, 75121530)
        self.assertEqual(df.loc[PAGEID_OCONNOR].viaf, 17227472)
        self.assertEqual(df.loc[PAGEID_DOEBLIN].viaf, 51688166)


class TestArticleLength(unittest.TestCase):

    def test_length(self):
        pageids = [PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR]
        df = public_domain_rank.features.article_lengths(pageids)
        self.assertGreater(df.loc[PAGEID_EINSTEIN].article_length, 0)
        self.assertGreater(df.loc[PAGEID_OCONNOR].article_length, 0)
        self.assertGreater(df.loc[PAGEID_DOEBLIN].article_length, 0)
        # Albert Einstein is a very long article
        self.assertGreater(df.loc[PAGEID_EINSTEIN].article_length, df.loc[PAGEID_OCONNOR].article_length)
        self.assertGreater(df.loc[PAGEID_EINSTEIN].article_length, df.loc[PAGEID_DOEBLIN].article_length)


class TestRevisions(unittest.TestCase):

    def test_last_revisions(self):
        pageids = [PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR, PAGEID_WRIGHT]
        df = public_domain_rank.features.last_revision(pageids)
        # Anticipate that Albert Einstein received a revision most recently
        einstein_seconds = df.loc[PAGEID_EINSTEIN].revision_timestamp
        wright_seconds = df.loc[PAGEID_WRIGHT].revision_timestamp
        doeblin_seconds = df.loc[PAGEID_DOEBLIN].revision_timestamp
        self.assertGreater(einstein_seconds, wright_seconds)
        self.assertGreater(einstein_seconds, doeblin_seconds)

    def test_revisions_per_day(self):
        pageids = [PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR, PAGEID_WRIGHT]
        df = public_domain_rank.features.revisions_per_day(pageids)
        # Einstein should receive more than 10 revisions a month
        einstein_rate = df.loc[PAGEID_EINSTEIN].revisions_per_day
        self.assertGreater(einstein_rate, 10 / 30)

        # Anticipate that Albert Einstein receives more revisions per day
        einstein_rate = df.loc[PAGEID_EINSTEIN].revisions_per_day
        wright_rate = df.loc[PAGEID_WRIGHT].revisions_per_day
        doeblin_rate = df.loc[PAGEID_DOEBLIN].revisions_per_day
        self.assertGreater(einstein_rate, wright_rate)
        self.assertGreater(einstein_rate, doeblin_rate)

    def test_revisions_per_day_cutoff(self):
        csv_fn = os.path.expanduser(public_domain_rank.config['data']['revision-metadata'])
        df_revs = pd.read_csv(csv_fn, sep='\t').set_index('pageid')
        today_timestamp = max(df_revs.first_revision) + 1
        pageid_recent = sorted(df_revs[df_revs.first_revision > (today_timestamp - 3600*23)].index).pop()

        pageids = [pageid_recent, PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR, PAGEID_WRIGHT]
        df = public_domain_rank.features.revisions_per_day(pageids, cutoff_days=20)
        self.assertGreater(2, df.loc[pageid_recent].revisions_per_day)

    def test_article_age(self):
        # Suspect that Einstein was created first
        pageids = [PAGEID_DOEBLIN, PAGEID_EINSTEIN, PAGEID_OCONNOR, PAGEID_WRIGHT]
        df = public_domain_rank.features.article_age(pageids)
        einstein_age = df.loc[PAGEID_EINSTEIN].article_age
        wright_age = df.loc[PAGEID_WRIGHT].article_age
        doeblin_age = df.loc[PAGEID_DOEBLIN].article_age
        self.assertGreater(einstein_age, wright_age)
        self.assertGreater(einstein_age, doeblin_age)


class TestOnlineBooksPage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.onlinebookspage()

    def test_lenin_trotsky(self):
        # these were notably absent in an earlier run despite many digital editions existing
        df = self.df
        self.assertGreater(df.loc[PAGEID_LENIN].obp_n_titles, 0)
        self.assertGreater(df.loc[PAGEID_TROTSKY].obp_n_titles, 0)

    def test_authors(self):
        df = self.df
        pageids = [PAGEID_ORWELL, PAGEID_WOOLF]
        for pageid in pageids:
            self.assertGreater(df.loc[pageid].obp_n_titles, 0)


class TestDigitalEditions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.digital_editions()

    def test_digital_editions_present(self):
        df = self.df
        authors_with_digital_editions = {21588568, 82142, 937059, 4447098, 36649366, 8541973, 12465552,
                                         165000, 2773024, 29412729, 533114, 931186, 3808626, 1098693,
                                         9724, 21043919, 4128471, 25728327, 2781794, 2495475, 1184453,
                                         14768003, 1064833, 1780114, 578724, 9709394, 1709823}
        missing = authors_with_digital_editions - set(df[df.obp_digital_editions > 0].index)
        assert not missing

    def test_digital_editions_absent(self):
        df = self.df
        authors_without_digital_editions = {931138, 422139, 251235, 787671, 174664,
                                            35199838, 1423328, 2370772, 617096, 28915030,
                                            8613601, 24503, 29528265}
        missing = authors_without_digital_editions - set(df[df.obp_digital_editions == 0].index)
        assert not missing


class TestDeathDecade(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.death_decade_dummies(1910, 1950)

    def test_death_decade_dummies(self):
        df = self.df
        # each row can only have one indicator that equals 1
        self.assertEqual(len(df), df.values.sum())

    def test_death_decade_authors(self):
        df = self.df
        self.assertEqual(df.loc[PAGEID_ORWELL].death_year_gte_1950, 1)
        self.assertEqual(df.loc[PAGEID_ORWELL].death_year_1940s, 0)
        self.assertEqual(df.loc[PAGEID_WOOLF].death_year_1940s, 1)
        self.assertEqual(df.loc[PAGEID_WOOLF].death_year_1930s, 0)
        self.assertEqual(df.loc[PAGEID_LADY_MORGAN].death_year_lt_1910, 1)
        self.assertEqual(df.loc[PAGEID_LADY_MORGAN].death_year_1940s, 0)


class TestDeathYear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.death_year()

    def test_death_year(self):
        df = self.df
        self.assertEqual(df.loc[PAGEID_LADY_MORGAN].death_year, 1859)


class TestCategoryDummiesOfInterest(unittest.TestCase):

    categories_bibliographic = [
        'category__Wikipedia_articles_with_BNF_identifiers',
        'category__Wikipedia_articles_with_GND_identifiers',
        'category__Wikipedia_articles_with_ISNI_identifiers',
        'category__Wikipedia_articles_with_LCCN_identifiers',
        'category__Wikipedia_articles_with_VIAF_identifiers',
    ]

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.category_dummies_of_interest()

    def test_category_featured_articles(self):
        # very few articles are given Featured status
        df = self.df
        self.assertLess(df['category__Featured_articles'].mean(), 0.2)

    def test_alfred_doeblin(self):
        df = self.df
        for cat in self.categories_bibliographic:
            self.assertEqual(getattr(df.loc[PAGEID_DOEBLIN], cat), 1)

    def test_richard_wright(self):
        df = self.df
        for cat in self.categories_bibliographic:
            self.assertEqual(getattr(df.loc[PAGEID_WRIGHT], cat), 1)


class TestPagecounts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = public_domain_rank.features.pagecounts()

    def test_pagecount(self):
        df = self.df
        pageids = [PAGEID_ORWELL, PAGEID_WOOLF, PAGEID_DOEBLIN,
                   PAGEID_EINSTEIN, PAGEID_OCONNOR, PAGEID_WRIGHT,
                   PAGEID_LENIN, PAGEID_TROTSKY]
        for pageid in pageids:
            self.assertGreater(df.loc[pageid].values[0], 0)
        self.assertGreater(df.loc[PAGEID_OCONNOR].values[0], df.loc[PAGEID_DOEBLIN].values[0])
        self.assertGreater(df.loc[PAGEID_EINSTEIN].values[0], df.loc[PAGEID_DOEBLIN].values[0])
