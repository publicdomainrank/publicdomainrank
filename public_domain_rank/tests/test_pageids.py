import unittest

from public_domain_rank import features


class TestPageidsCanonical(unittest.TestCase):

    df = features.pageids_canonical()

    def test_no_redirects(self):
        df = self.df
        self.assertIn(96429, df.index)  # Flannery O'Connor
        self.assertNotIn(16073500, df.index)  # A redirect that should be absent
