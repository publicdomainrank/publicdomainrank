# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html

from scrapy.item import Item, Field


class Author(Item):
    author = Field()
    link = Field()
    num_titles = Field()
    page = Field()
    wikipedia_link = Field()
