"""
Script to extract page titles and page texts to files from an XML dump.

"""

import os
import tempfile

from mw import xml_dump

import public_domain_rank
import public_domain_rank.features

# we have to preload this to avoid diving into all the revisions of every page
_PAGEIDS_OF_INTEREST = set(public_domain_rank.features.pageids_canonical().pageid)


def page_info(dump, path):
    """Extract pageid, namespace, title, revsion timestamp, and text.

    Intended to be used with ``mw.xml_dump`` and with an enwiki "articles" xml
    dump.
    """
    for page in dump:
        if page.namespace == 0 and page.id in _PAGEIDS_OF_INTEREST:
            revisions = [rev for rev in page]
            assert len(revisions) == 1
            revision = revisions.pop()  # there should only be one revision
            # have to make a copy of this revision thing, it has trouble being pickled
            yield (page.id, page.namespace, page.title, str(revision.timestamp), str(revision.text))
        else:
            # if not of interest indicate with empty tuple
            yield tuple()


def extract_pages(output_path):
    articles_dump_fn = os.path.expanduser(public_domain_rank.config['data']['articles'])
    print("number of pageids of interest: {}".format(len(_PAGEIDS_OF_INTEREST)))
    counter = 0
    pageid_title_timestamp_length = []
    for i, page_info_tuple in enumerate(xml_dump.map([articles_dump_fn], page_info)):
        # empty tuple indicates not of interest
        if page_info_tuple is tuple():
            continue
        pageid, namespace, title, timestamp, text = page_info_tuple
        article_length = len(text)
        pageid_title_timestamp_length.append((pageid, title, timestamp, article_length))
        text_fn = os.path.join(output_path, '{}.txt'.format(pageid))
        with open(text_fn, 'w') as f:
            f.write(text)
        counter += 1
        if counter % 1e4 == 0:
            print("extracted {} pages".format(counter))
    pageid_title_output_fn = os.path.join(output_path, 'pageid-title-timestamp-length.csv')
    with open(pageid_title_output_fn, 'w') as f:
        # write header
        f.write('pageid\ttitle\trevision_timestamp\tarticle_length\n'.format(pageid, title, timestamp))
        for pageid, title, timestamp, article_length in pageid_title_timestamp_length:
            f.write('{}\t{}\t{}\t{}\n'.format(pageid, title, timestamp, article_length))
    print("finished extracting. extracted {} pages".format(counter))

if __name__ == '__main__':
    tempdir = tempfile.mkdtemp()
    print("extracting data into {}".format(tempdir))
    extract_pages(tempdir)
