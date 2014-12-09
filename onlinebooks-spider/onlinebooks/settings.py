# Scrapy settings for onlinebooks project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#

BOT_NAME = 'onlinebooks'

SPIDER_MODULES = ['onlinebooks.spiders']
NEWSPIDER_MODULE = 'onlinebooks.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'onlinebooks (+http://www.yourdomain.com)'

AUTOTHROTTLE_ENABLED = True
