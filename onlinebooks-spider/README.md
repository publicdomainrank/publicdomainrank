The Online Books Page spider
============================

A spider for collecting all author records from The Online Books Page is found
in``onlinebooks/spiders/``

Prerequisites
-------------

- Python 2.7
- scrapy

Quickstart
----------

    scrapy crawl onlinebooks_authors -t json -o authors.json 

Scraping with persistence
-------------------------

To start a spider with persistence supported enabled, run it like this:

    scrapy crawl onlinebooks_authors -t json -o crawls/authors-1.json -s JOBDIR=crawls/authors-1 --logfile=crawls/onlinebooks-1.log

Then, you can stop the spider safely at any time (by pressing Ctrl-C or sending
a signal), and resume it later by issuing the same command.

    scrapy crawl onlinebooks_authors -t json -o crawls/authors-1-resumed.json -s JOBDIR=crawls/authors-1 --logfile=crawls/onlinebooks-1-resumed.log
