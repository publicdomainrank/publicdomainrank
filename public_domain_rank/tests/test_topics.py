import unittest

import public_domain_rank.features as features
import public_domain_rank.topics as topics


def _top_topics_have_word(word, top_topics, top_words):
    found_word = False
    for top_topic in top_topics:
        if word in top_words[top_topic]:
            found_word = True
    return found_word


class TestTopicWords(unittest.TestCase):

    doctopic = topics.theta()
    top_words = topics.top_words_tfidf(topics.ntw())

    def test_pageids(self):
        # we need a topic distribution for every article
        doctopic = self.doctopic
        missing_pageids = set(features.pageids_canonical()['pageid']) - set(doctopic.index)
        self.assertEquals(missing_pageids, set())

    def test_alfred_doeblin(self):
        pageid = 896457
        doctopic, top_words = self.doctopic, self.top_words
        self.assertIn(pageid, doctopic.index)
        top_topics = doctopic.loc[pageid].values.argsort()[::-1][0:5]
        word = 'writer'
        self.assertTrue(_top_topics_have_word(word, top_topics, top_words))

    def test_richard_wright(self):
        pageid = 43153
        doctopic, top_words = self.doctopic, self.top_words
        self.assertIn(pageid, doctopic.index)
        top_topics = doctopic.loc[pageid].values.argsort()[::-1][0:5]
        word = 'writer'
        self.assertTrue(_top_topics_have_word(word, top_topics, top_words))

    def test_flannery_oconnor(self):
        pageid = 96429
        doctopic, top_words = self.doctopic, self.top_words
        self.assertIn(pageid, doctopic.index)
        top_topics = doctopic.loc[pageid].values.argsort()[::-1][0:5]
        word = 'writer'
        self.assertTrue(_top_topics_have_word(word, top_topics, top_words))

    def test_albert_einstein(self):
        pageid = 736
        doctopic, top_words = self.doctopic, self.top_words
        self.assertIn(pageid, doctopic.index)
        top_topic = doctopic.loc[pageid].values.argmax()
        self.assertNotIn('literatur', top_words[top_topic])
        self.assertIn('scienc', top_words[top_topic])
