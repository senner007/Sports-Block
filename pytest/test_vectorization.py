   # The code to test
import sys
sys.path.append('..')

from csv_data import csv_list_to_list
from vectorization import to_lower
from vectorization import remove_dash
from vectorization import split_included_specials
from vectorization import replace_tournament
from vectorization import replace_countries
from vectorization import replace_weekday
from vectorization import replace_digits
from vectorization import replace_finals
from vectorization import replace_nationality
from vectorization import vect_layer_2_text
from vectorization import vectorize_layer
from vectorization import standardize
from static_data import tournaments
from static_data import weekdays
from static_data import non_alpha
from static_data import word_generalization

countries = csv_list_to_list('resources/countries.csv')
nationalities = csv_list_to_list('resources/nat2.csv')

words_train_vocab = ["formel", "fodbold", "pokalen", "-", "mål", "duel"]

arrs = [
    to_lower, 
    remove_dash, 
    split_included_specials, 
    replace_tournament(tournaments),
    replace_countries(countries), 
    replace_weekday(weekdays), 
    replace_finals,
    replace_nationality(nationalities),
    replace_digits
]

s = standardize(arrs)

words_train_vocab.extend(word_generalization)
words_train_vocab.extend(non_alpha)

max_features = 5700
sequence_length = 100

vectorized_layer = vectorize_layer(max_features, sequence_length, s)

text_ds = vectorized_layer.adapt(words_train_vocab)
vect_vocab = vectorized_layer.get_vocabulary()

vectorization_tests = {
    "Fodbold tour-de-france-pokalen" : ['fodbold', 'xtournament', 'pokalen'],
    "Fodbold Tour de-france pokalen" : ['fodbold', 'xtournament', 'pokalen'],
    "Danmark England Tyskland" : ['xland', 'xland','xland'],
    "albaniens indonesien portugal" : ['xland', 'xland','xland'],
    "majoren Majoren Majorens majorens" : ['xtournament', 'xtournament', 'xtournament', 'xtournament'],
    "pga PGA pga-turneringerne pga turneringens PGA-turneringen pga-turneringer turnering turneringer" : ['xtournament', 'xtournament', 'xtournament', 'xtournament', 'xtournament', 'xtournament', 'xtournament', 'xtournament'],
    "Formel-1-grand prixet grandprixet formel-1-grandprixet"  : ['formel', 'xnumber', 'xtournament', 'xtournament', 'formel', 'xnumber', 'xtournament'],
    "2 fodbold 1938 fodbold 5 fodbold 2020": ['xnumber', 'fodbold','xyear', 'fodbold', 'xnumber', 'fodbold', 'xyear'],
    "fodbold danske dansker engelske englænder englænderen" : ["fodbold", "xnationality",  "xnationality",  "xnationality",  "xnationality", "xnationality"],
    "fodbold-pokalen": ["fodbold", "pokalen"],
    "fodbold-pokalen": ["fodbold", "pokalen"],
    "fodbold  -  pokalen": ["fodbold", "pokalen"],
    "dansk danskermål danskerduel": ["xnationality", "xnationality", "mål", "xnationality", "duel"],
    "kroatisk vietnamesisk indonesisk tysk engelsk" : ["xnationality", "xnationality", "xnationality",  "xnationality"]
}

def test_vectorization():
    for v in vectorization_tests:
        actual = vect_layer_2_text(vectorized_layer([v]), vect_vocab)
        expected = vectorization_tests[v]
        print(actual)
        assert all([a == b for a, b in zip(list(actual), list(expected))])