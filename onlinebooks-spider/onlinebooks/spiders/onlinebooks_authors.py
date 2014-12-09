import re

from scrapy.http import Request
from scrapy.spider import BaseSpider
from scrapy.selector import Selector

from onlinebooks.items import Author


class OnlineBooksSpider(BaseSpider):
    name = "onlinebooks_authors"
    allowed_domains = ["onlinebooks.library.upenn.edu"]
    start_urls = [r"http://onlinebooks.library.upenn.edu/webbin/book/authorstart?A"]

    def parse(self, response):
        hxs = Selector(response)
        authorlist = hxs.xpath('//body/ul/li')
        for a in authorlist:
            item = Author()
            item['author'] = a.xpath('a/text()').extract()[0]
            num_titles_str = a.xpath('text()').extract()[0]
            item['num_titles'] = int(re.findall(r'([0-9]+) title', num_titles_str)[0])
            item['link'] = a.xpath('a/@href').extract()[0]
            request = Request(item['link'], callback=self.parse_item)
            request.meta['item'] = item
            yield request

        # get next page
        navlinks = hxs.xpath('//body/p/a')
        for link in navlinks:
            name = link.xpath('text()').extract()[0]
            url = link.xpath('@href').extract()[0]
            if re.match(r'next', name):
                yield Request(url, callback=self.parse)

    def parse_item(self, response):
        self.state['items_count'] = self.state.get('items_count', 0) + 1
        item = response.meta['item']
        hxs = Selector(response)
        links = hxs.xpath('//a/@href')
        for link in links:
            href = link.extract().strip()
            if 'wikipedia' in href:
                item['wikipedia_link'] = href
                break  # just take the first one
        item['page'] = hxs.xpath('//body').extract()
        return item
