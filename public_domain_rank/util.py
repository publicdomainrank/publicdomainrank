import json
import os
import re
import urllib.parse

import numpy as np
import requests
import xmltodict

WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"
LOC_SEARCH_ENDPOINT = 'http://id.loc.gov/search/'
LOC_NAME_ENDPOINT = 'http://id.loc.gov/authorities/names/'
VIAF_LOC_NAME_ENDPOINT = 'http://viaf.org/viaf/sourceID/LC|n+'


def pageid_from_title(title):
    """Fetch pageid for a Wikipedia article title from Mediawiki API"""
    params = dict(action="query", prop="info", titles=title, format="json")
    return json.loads(requests.get(WIKIPEDIA_ENDPOINT, params=params).text)['query']


def extract_death_year(s):
    """Extract year of death from a Library-of-Congress-style author name.

    For example, "Grey, Zane, 1872-1939" -> 1939

    """
    try:
        death_year_str = re.findall(r'-([0-9][0-9][0-9][0-9])', s).pop()
    except IndexError:
        return float('nan')
    try:
        death_year = int(death_year_str)
    except ValueError:
        return float('nan')
    return death_year


def get_loc_name_id(author_name):
    """Get a possible Library of Congress name identifier from an author's name"""
    params = [('q', author_name),
              ('q', 'cs:http://id.loc.gov/authorities/names'),
              ('format', 'xml')]
    params_encoded = urllib.parse.urlencode(params, doseq=True)
    data = xmltodict.parse(requests.get(LOC_SEARCH_ENDPOINT + '?' + params_encoded).content)
    try:
        results = data['search:response']['search:result']
        results.reverse()
        result = results.pop()
    except AttributeError:
        # no results
        return float('nan')
    except KeyError:
        # no results
        return float('nan')
    except IndexError:
        # no results
        return float('nan')

    try:
        name_id = int(os.path.splitext(result['@uri'].split('/').pop())[0].lstrip('n'))
        score = float(result['@score'])
        confidence = float(result['@confidence'])
    except ValueError:
        return float('nan')

    if score > 0 and confidence > 0:
        return name_id
    else:
        return float('nan')


def get_viaf_id(author_name):
    """Get a possible VIAF identifier for a given author's name"""
    loc_name_id = get_loc_name_id(author_name)
    if np.isnan(loc_name_id):
        return float('nan')
    # set stream to True since we don't care about the request
    url = VIAF_LOC_NAME_ENDPOINT + str(loc_name_id)
    response = requests.get(url, stream=True)
    print("found loc name id: ", author_name, loc_name_id, url)
    # the VIAF server will redirect, get VIAF number from the url
    if len(response.history) == 0:
        print("Warning: expected a redirect to VIAF url for ", author_name)
        return float('nan')
    viaf = int(re.findall(r'[0-9]{3,}', response.url).pop())
    return viaf
