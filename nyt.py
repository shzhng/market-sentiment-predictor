#!/usr/bin/env python2
# File to interact with the New York Times API and get the necessary data


import json
import math
import requests
from datetime import date, timedelta

NYT_URI = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'

def getPreOpenNews(day):
    params = {}
    params['begin_date'] = day
    params['end_date'] = day
    params['sort'] = 'oldest'
    params['fl'] = 'headline,lead_paragraph,abstract,pub_date'
    params['api-key'] = 'a4e3bf0b48abf9aff1cc6acd5be3633f:18:70145062'
    params['fq'] = 'news_desk:("Business")'

    news = []
    resp = requests.get(NYT_URI, params=params)
    if not resp.status_code == requests.codes.ok:
        return
    resp = resp.json()
    hits = resp[u'response'][u'meta'][u'hits']
    docs = resp[u'response'][u'docs']
    print hits
    for i in range(1, math.trunc(hits / 10) + 1):
        timeFields = docs[-1][u'pub_date'].split('T')[1].split(':')
        if int(timeFields[0]) >= 8 and int(timeFields[1]) >= 1:
            for j in range(len(docs)):
                timeFields = docs[j][u'pub_date'].split('T')[1].split(':')
                if int(timeFields[0]) >= 8 and int(timeFields[1]) >= 1:
                    news.append(docs[j])
            break
        else:
            news.extend(docs)

        # Get next page
        params['page'] = i
        resp = requests.get(NYT_URI, params=params)
        if not resp.status_code == requests.codes.ok:
            print 'HTTP request returned with status code:', resp.status_code
            break
        resp = resp.json()
        docs = resp[u'response'][u'docs']
        if len(docs) <= 0:
            break

    return news

def getSentimentByDay(day):
    news = getPreOpenNews(day)

    if news is None:
        print "Error occurred while grabbing the sentiment for the day:", day
        return

    print json.dumps(news, indent=4, separators=(',', ': '))


if __name__=='__main__':
    today = date.today()
    yeardelta = timedelta(days=365)
    daydelta = timedelta(days=1)
    d = today - yeardelta
    while d <= today:
        # Only get weekdays
        if d.weekday() < 5:
            datestr = d.strftime('%Y%m%d')
            print datestr
            with open(datestr, 'w') as out:
                news = getPreOpenNews(datestr)
                out.write(json.dumps(news, indent=4, separators=(',', ': ')))

        d += daydelta
