"""Gets a bunch of names from sec.gov"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import requests
from lxml import html


def get_names(path='names.txt'):
    """Gets the list of names. Checks to see if we've already done
    the gross web stuff, if so reads it and if not does the stuff.

    Args:
        path: where the files are.

    Returns:
        list of strings.
    """
    if not os.path.exists(path):
        logging.info('data not found, downloading.')
        url = 'https://www.sec.gov/rules/other/4-460list.htm'
        r = requests.get(url)
        tree = html.fromstring(r.content)
        names = tree.xpath('//tr/td[2]')
        names = [name.text.strip()
                 for name in names if name.text is not None]

        url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?industry=Technology&render=download'
        r = requests.get(url)
        newnames = [line.strip().split(',')[1].replace('"', '')
                    for line in r.text.split('\n')[:-1]]
        names.extend(newnames)
        with open(path, 'w') as txtfile:
            txtfile.write('\n'.join(names))
    else:
        with open(path, 'r') as txtfile:
            names = txtfile.read().split('\n')
    return names


if __name__ == '__main__':
    print(len(get_names()))
