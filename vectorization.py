from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import numpy as np
from static_data import non_alpha
from static_data import weekdays
from static_data import tournaments
from csv_data import csv_list_to_list
nationalities = csv_list_to_list('resources/nationalities.csv')
countries = csv_list_to_list('resources/countries.csv')

def increment(x):
    return x + 1

# TODO : evt indikere hvilke navneord der starte med stort bogstav(egenavne), evt. lave et opslag for at undersøge ordklasse for det første ord i sætningen 

def to_lower(word):
    return tf.strings.lower(word, encoding='utf-8')

def remove_dash(word):
  return tf.strings.regex_replace(word, '-', ' ')

def replace_finals(word):
  r = r'\w*finale'
  return tf.strings.regex_replace(word, pattern=r, rewrite="xfinale")

def split_included_specials(word):

    new_str = tf.strings.regex_replace(word, pattern=r'([^a-zæøåñäöîçíãúéïèüáëó0-9\s])', rewrite=r' \1 ', replace_global=True)
    new_str = tf.strings.regex_replace(new_str, pattern=r'([»«_,?\'\"])', rewrite=r'', replace_global=True)
    return new_str

def replace_digits(word):

    new_str = word
    new_str = tf.strings.regex_replace(new_str, pattern=r'(?:18|19|20)\d{2}', rewrite=r'xyear')
    new_str = tf.strings.regex_replace(new_str, pattern=r'\d+', rewrite=r'xnumber', replace_global=True)
    new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:to|tre|fire|fem|seks|syv|otte|ni|ti)\b', rewrite=r'xnumber_multiple', replace_global=True)
    new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:anden|tredje|fjerde|femte|sjette|syvende|ottende|niende|tiende)(?:-)?', rewrite=r'xnumber_multiple', replace_global=True)

    return new_str


def replace_countries(countries):
    def replace_countries(word):

        new_str = word
        for sign in countries:
            r = "\\b(?:nord|syd|øst|vest)?" + sign + "s?\\b"
            new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xland")

        return new_str
    return replace_countries


def replace_tournament(tournaments):
    def replace_tournament(word):

        new_str = word
        for sign in tournaments:
            # r = "\\b(?:" + sign + "|turnering)\\w*(?:(?:(\\sturnering)|-turnering)\\w*)?" + "\\b" #https://regex101.com/r/5qk5nk/1
            # r = "(?:\\b" + sign + "(?: turnering|-turnering)?)(?:et|er|en|erne)?\\b"  #https://regex101.com/r/L6tEaM/1
            r = "\\b" + sign + "(?: turnering|-turnering)?(?:s|et|er|en|ens|erne)?\\b"
            new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xtournament")

        return new_str
    return replace_tournament

def replace_nationality(nationalities):
    def replace_nationality(word):
        new_str = word
        for sign in nationalities:
            r = "\\b(?:nord|syd|øst|vest)?" + sign + "(?:eren|erne|ere|ne|en|er|r|e|isk)?" + "(\w*)\\b" #https://regex101.com/r/G6LBoR/1
            new_str = tf.strings.regex_replace(new_str, pattern=r,  rewrite=r'xnationality \1')

        return new_str
    return replace_nationality

def replace_weekday(weekdays):
        def replace_weekday(word):
            new_str = word
            for sign in weekdays:
                r = "\\b" + sign + "(?:s|en)?\\b"  # https://regex101.com/r/t7KC9v/1
                new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xweekday")
            return new_str
        return replace_weekday


# TODO : test hvilke standarization funktioner giver bedre resultater 

def standardize(func_arr):
    def iterate_funcs(x):
        val = x
        for f in func_arr:
            val = f(val)
        return val
    return iterate_funcs

def vect_layer_2_text(vect, vect_vocab):
    return np.array([vect_vocab[x] for x in np.trim_zeros(np.squeeze(vect.numpy()))])
 
weekdays_func = replace_weekday(weekdays)

# vectorization_formatters = [
#     to_lower, 
#     remove_dash, 
#     split_included_specials, 
#     replace_tournament(tournaments),
#     replace_countries(countries), 
#     replace_weekday(weekdays), 
#     replace_finals,
#     replace_nationality(nationalities),
#     replace_digits
# ]

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data): 
    x = to_lower(input_data)
    x = remove_dash(x)
    x = split_included_specials(x)
    x = replace_tournament(tournaments)(x)
    x = replace_countries(countries)(x)
    x = replace_weekday(weekdays)(x) 
    x = replace_finals(x)
    x = replace_nationality(nationalities)(x)
    x = replace_digits(x)
    return x

@tf.keras.utils.register_keras_serializable()
def vectorize_layer(max_features, sequence_length, standardization):
    return TextVectorization(
    standardize=standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length)

