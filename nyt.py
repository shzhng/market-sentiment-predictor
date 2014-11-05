import json
import math
import requests

NYT_URI = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'

def getPreOpenNews(day):
    params = {}
    params['begin_date'] = day
    params['end_date'] = day
    params['sort'] = 'oldest'
    params['fl'] = 'headline,lead_paragraph,abstract,pub_date,snippet,print_page,keywords'
    params['api-key'] = 'a4e3bf0b48abf9aff1cc6acd5be3633f:18:70145062'

    news = []
    resp = requests.get(NYT_URI, params=params)
    if not resp.status_code == requests.codes.ok:
        return
    resp = resp.json()
    hits = resp[u'response'][u'meta'][u'hits']
    docs = resp[u'response'][u'docs']
    print hits
    for i in range(1, math.trunc(hits / 10) + 1):
        if docs[-1][u'pub_date']: # TODO less than market open
            news.extend(docs)
        else: # TODO only add up as far as market open
            for j in range(len(docs)):
                if docs[j][u'pub_date']: # TODO is less than market open
                    news.append(docs[j])

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

    print json.dumps(news, indent=4, separators=(',', ': '))

if __name__=='__main__':
    getSentimentByDay('20141104') # TEST
