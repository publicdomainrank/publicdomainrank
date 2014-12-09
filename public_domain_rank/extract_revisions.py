"""Script to extract selected information about articles' revisions."""

import os
import tempfile
import time

import mw.xml_dump

import public_domain_rank
import public_domain_rank.features

# we have to preload this to avoid diving into all the revisions of every page
_PAGEIDS_OF_INTEREST = set(public_domain_rank.features.pageids_canonical().pageid)


def revision_info(dump, path):
    """Extract number of revisions and the timestamp of the first revision"""
    for page in dump:
        if page.namespace == 0 and page.id in _PAGEIDS_OF_INTEREST:
            timestamps = [rev.timestamp.unix() for rev in page]
            assert len(timestamps) > 0
            first_revision = min(timestamps)
            yield (page.id, page.namespace, len(timestamps), first_revision)
        else:
            yield tuple()


def extract_revision_metadata(output_path):
    history_dump_fn = os.path.expanduser(public_domain_rank.config['data']['history'])
    print("number of pageids of interest: {}".format(len(_PAGEIDS_OF_INTEREST)))
    counter = 0
    t0 = time.time()
    revision_metadata_fn = os.path.join(output_path, 'revision-metadata.csv')
    f = open(revision_metadata_fn, 'w')
    f.write('pageid\tn_revisions\tfirst_revision\n')
    for i, revision_metadata in enumerate(mw.xml_dump.map([history_dump_fn], revision_info)):
        if revision_metadata is tuple():
            continue
        pageid, namespace, n_revisions, first_revision = revision_metadata
        f.write('{}\t{}\t{}\n'.format(pageid, n_revisions, first_revision))
        counter += 1
        if i % 100 == 0:
            f.flush()
            rate = (i + 1) / (time.time() - t0)
            print("extracted history for {} pages, {} pages seen, {} pages/second".format(counter, i, rate))
    f.close()
    print("finished extracting history. extracted history for {} pages".format(counter))

if __name__ == '__main__':
    t0 = time.time()
    tempdir = tempfile.mkdtemp()
    print("extracting data into {}".format(tempdir))
    extract_revision_metadata(tempdir)
    print("extracted data into {}".format(tempdir))
    print("extracting duration: ", time.time() - t0)
