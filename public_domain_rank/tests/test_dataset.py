import unittest

import public_domain_rank


class TestDataset(unittest.TestCase):

    df = public_domain_rank.dataset(langlink_max_features=20, death_decade_start=1910, death_decade_stop=1950)

    def test_redirects_flannery_oconnor(self):
        df = self.df
        redirect_set = df.loc[96429].redirect_set
        n_redirect = df.loc[96429].n_redirect
        self.assertGreater(len(redirect_set), 0)
        self.assertEqual(len(redirect_set), n_redirect)
        self.assertIn("O'Connor,_Flannery", redirect_set)

    def test_pagecounts(self):
        df = self.df
        oconnor_pagecount = df.loc[96429]['pagecount']
        doeblin_pagecount = df.loc[896457]['pagecount']
        einstein_pagecount = df.loc[736]['pagecount']
        self.assertGreater(doeblin_pagecount, 0)
        self.assertGreater(oconnor_pagecount, 0)
        self.assertGreater(einstein_pagecount, 0)
        self.assertGreater(oconnor_pagecount, doeblin_pagecount)
        self.assertGreater(einstein_pagecount, doeblin_pagecount)
        self.assertGreater(einstein_pagecount, oconnor_pagecount)
