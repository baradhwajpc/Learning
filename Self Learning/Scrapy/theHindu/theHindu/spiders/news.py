# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from theHindu.items import ThehinduItem

class NewsSpider(CrawlSpider):
    name = 'news'
    allowed_domains = ['thehindu.com']
    start_urls = ['http://www.thehindu.com/news/national/',
                  'http://www.thehindu.com/news/international/',
                  'http://www.thehindu.com/news/states/',
                  'http://www.thehindu.com/news/cities/'
    ]
    
    rules = (
        Rule(LinkExtractor(allow=(), restrict_css=('.page-link',)),
             callback="parse_item",
             follow=False),)

    def parse_item(self, response):
        print('Processing..' + response.url)
        item_links = response.css('.story-card-news > h3 > a::attr(href)').extract()
        #item_links = response.xpath('//h3/text() > a::attr(href)').extract()
        for a in item_links:
            yield scrapy.Request(a, callback=self.parse_detail_page)

    def parse_detail_page(self, response):
        title = response.css('h1::text').extract()[0].strip()
        category = response.css('.article-exclusive > a::text').extract()[0].strip()
        
        item = ThehinduItem() 
        item['title'] = title
        item['category'] = category
        item['url'] = response.url
        yield item