   # The code to test
import sys
sys.path.append('..')

from csv_data import csv_to_list
from vectorization import to_lower
from vectorization import split_dash
from vectorization import split_specials
from vectorization import replace_tournament
from vectorization import replace_countries
from vectorization import replace_weekday
from vectorization import replace_digits
from vectorization import vect_layer_2_text
from vectorization import vectorize_layer
from vectorization import standardize
from static_data import tournaments
from static_data import weekdays
from static_data import non_alpha
from static_data import word_generalization

countries = csv_to_list('countries.csv')

words_train_vocab = ["fodbold", "pokalen", "-"]

arrs = [
    to_lower, 
    split_dash, 
    split_specials, 
    replace_tournament(tournaments), 
    replace_countries(countries), 
    replace_weekday(weekdays), 
    replace_digits
]

s = standardize(arrs)

words_train_vocab.extend(word_generalization)
words_train_vocab.extend(non_alpha)

# Model constants.
max_features = 5700
sequence_length = 100

vectorized_layer = vectorize_layer(max_features, sequence_length, s)

text_ds = vectorized_layer.adapt(words_train_vocab)
vect_vocab = vectorized_layer.get_vocabulary()

vectorization_tests = {
    "Fodbold tour-de-france-pokalen" : ['fodbold', 'x_tournament', 'pokalen'],
    "Fodbold Tour de-france pokalen" : ['fodbold', 'x_tournament', 'pokalen'],
    "Danmark" : ['x_land'],
    "albanien" : ['x_land'],
    "uefa Uefa UEFA uefas Uefas" : ['x_tournament', 'x_tournament', 'x_tournament', 'x_tournament', 'x_tournament'],
    "majoren Majoren Majorens majorens" : ['x_tournament', 'x_tournament', 'x_tournament', 'x_tournament'],
    "pga PGA pga-turneringerne pga turneringens PGA-turneringen pga-turneringer turnering turneringer" : ['x_tournament', 'x_tournament', 'x_tournament', 'x_tournament', 'x_tournament', 'x_tournament', 'x_tournament', 'x_tournament']
    


}

def test_vectorization():
    for v in vectorization_tests:
        actual = vect_layer_2_text(vectorized_layer([v]), vect_vocab)
        expected = vectorization_tests[v]
        assert all([a == b for a, b in zip(list(actual), list(expected))])



# print (vect_layer_2_text(vectorized_layer(["Danske nordjyderne københavnerne, midtjyderne København Fodbold . Mandags og Tirsdag Tysklands Tour de France-Pokalen Superligaen turnering Superliga-turneringen og Wimbledon og World-cup turnering"]), vect_vocab))

# print (vect_layer_2_text(vectorize_layer(["Fodbold tour-de-france-pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold Tour de-france pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold mægtige-ryder cup"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold ryder cup-drama"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold Ryder Cup drama"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold ryder-cup-pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold UFC-pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold ufc pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Fodbold NFL-pokalen"])))
# print (vect_layer_2_text(vectorize_layer(["Ruslands Ol-Komité"])))
# print (vect_layer_2_text(vectorize_layer(["Ruslands Ol-Komité"])))

# print (vect_layer_2_text(vectorize_layer(["Ruslands Olympiske Komité"])))
# print (vect_layer_2_text(vectorize_layer(["Vandt Arctic Open"])))
# print (vect_layer_2_text(vectorize_layer(["Vandt Arctic-open"])))
# print (vect_layer_2_text(vectorize_layer(["Vandt Super-Bowl"])))
# print (vect_layer_2_text(vectorize_layer(["Vandt Super-bowl-mestre"])))
# print (vect_layer_2_text(vectorize_layer(["ATP-turneringen er i gang"])))
# print (vect_layer_2_text(vectorize_layer(["Super-G-løb"])))



# print (vect_layer_2_text(vectorize_layer(["Færøerne vinder"])))
# print (vect_layer_2_text(vectorize_layer(["færøernes hold"])))
# print (vect_layer_2_text(vectorize_layer(["østrigs hold"])))
# print (vect_layer_2_text(vectorize_layer(["Østrig hold"])))
# print (vect_layer_2_text(vectorize_layer(["Tysklands hold"])))


# print (vect_layer_2_text(vectorize_layer(["færøernes hold"])))


# print (vect_layer_2_text(vectorize_layer(["det albansk hold"])))
# print (vect_layer_2_text(vectorize_layer(["angolanske vindere"])))
# print (vect_layer_2_text(vectorize_layer(["danskerne vinder"])))
# print (vect_layer_2_text(vectorize_layer(["de danske vindere"])))
# print (vect_layer_2_text(vectorize_layer(["det engelske hold"])))
# print (vect_layer_2_text(vectorize_layer(["england vinder"])))
# print (vect_layer_2_text(vectorize_layer(["England vinder"])))
# print (vect_layer_2_text(vectorize_layer(["I onsdags blev"])))