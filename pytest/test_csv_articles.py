
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
import collections
from csv_data import get_all_articles

sentences_total, labels_total = get_all_articles(shuffle=False)

allowed_characters = ['d', 't', 'ú', '3', 'î', '5', '.', 'n', '»', 'ç', '!', 'ó', '6', 'ü', 'x', '&', 'ø', '´', 'q', "'", 'ä', 'o', 'e', 'u', 'i', 'r', 'ë', ' ', 'c', '-', '0', 'g', 'w', 'z', 'í', 'æ', 'b', '#', 'è', '1', '(', 'ï', 'a', 'k', 'é', 'f', 'á', '4', 'p', '9', '/', ')', '+', '«', '7', 'ã', 'h', '"', 'l', 'å', 'v', ':', 'm', 'y', '2', 'ö', '?', 'ñ', 'j', ',', '8', 's']

def get_set_of_chars(str_arr):
    total_chars = []
    for sentence in sentences_total:
        chars = [x for x in sentence]
        total_chars.extend(chars) 

    return set(total_chars)

def test_allowed_chars_in_data():
    set_of_chars = get_set_of_chars(sentences_total)
    for c in set_of_chars:
        assert (c in allowed_characters)

def test_labels_format():
    for label in labels_total:
        assert (isinstance(label, np.int64) == True)
        assert (label == 0 or label == 1)

    