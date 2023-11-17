from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import numpy as np
from static_data import non_alpha
from static_data import weekdays
from static_data import tournaments
from csv_data import csv_list_to_list
import json
nationalities = csv_list_to_list('resources/nationalities.csv')
countries = csv_list_to_list('resources/countries.csv')

def increment(x):
    return x + 1

# TODO : evt indikere hvilke navneord der starte med stort bogstav(egenavne), evt. lave et opslag for at undersøge ordklasse for det første ord i sætningen 

def to_lower(word):
    return tf.strings.lower(word, encoding='utf-8')

def remove_dash(word):
  return tf.strings.regex_replace(word, '-', ' ')


def split_included_specials(word):

    new_str = tf.strings.regex_replace(word, pattern=r'([^a-zæøåñäöîçíãúéïèüáëó0-9\s])', rewrite=r' \1 ', replace_global=True)
    # new_str = tf.strings.regex_replace(new_str, pattern=r'([»«_,?\'\"])', rewrite=r'', replace_global=True)
    return new_str

# def replace_digits(word):

#     # new_str = word
#     # new_str = tf.strings.regex_replace(new_str, pattern=r'(?:18|19|20)\d{2}', rewrite=r'xyear')
#     # new_str = tf.strings.regex_replace(new_str, pattern=r'\d+', rewrite=r'xnumber', replace_global=True)
#     # new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:to|tre|fire|fem|seks|syv|otte|ni|ti)\b', rewrite=r'xnumber_multiple', replace_global=True)
#     # new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:anden|tredje|fjerde|femte|sjette|syvende|ottende|niende|tiende)(?:-)?', rewrite=r'xnumber_multiple', replace_global=True)

#     return new_str

# def replace_finals(word):
#   r = r'\w*finale'
#   return tf.strings.regex_replace(word, pattern=r, rewrite="xfinale")

# def replace_countries(countries):
#     def replace_countries(word):

#         new_str = word
#         for sign in countries:
#             r = "\\b(?:nord|syd|øst|vest)?" + sign + "s?\\b"
#             new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xland")

#         return new_str
#     return replace_countries


# def replace_tournament(tournaments):
#     def replace_tournament(word):

#         new_str = word
#         for sign in tournaments:
#             # r = "\\b(?:" + sign + "|turnering)\\w*(?:(?:(\\sturnering)|-turnering)\\w*)?" + "\\b" #https://regex101.com/r/5qk5nk/1
#             # r = "(?:\\b" + sign + "(?: turnering|-turnering)?)(?:et|er|en|erne)?\\b"  #https://regex101.com/r/L6tEaM/1
#             r = "\\b" + sign + "(?: turnering|-turnering)?(?:s|et|er|en|ens|erne)?\\b"
#             new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xtournament")

#         return new_str
#     return replace_tournament

# def replace_nationality(nationalities):
#     def replace_nationality(word):
#         new_str = word
#         for sign in nationalities:
#             r = "\\b(?:nord|syd|øst|vest)?" + sign + "(?:eren|erne|ere|ne|en|er|r|e|isk)?" + "(\w*)\\b" #https://regex101.com/r/G6LBoR/1
#             new_str = tf.strings.regex_replace(new_str, pattern=r,  rewrite=r'xnationality \1')

#         return new_str
#     return replace_nationality

# def replace_weekday(weekdays):
#         def replace_weekday(word):
#             new_str = word
#             for sign in weekdays:
#                 r = "\\b" + sign + "(?:s|en)?\\b"  # https://regex101.com/r/t7KC9v/1
#                 new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="xweekday")
#             return new_str
#         return replace_weekday



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


    new_str = word
    # new_str = tf.strings.regex_replace(new_str, pattern=r'(?:18|19|20)\d{2}', rewrite=r'xyear')
    # new_str = tf.strings.regex_replace(new_str, pattern=r'\d+', rewrite=r'xnumber', replace_global=True)
    # new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:to|tre|fire|fem|seks|syv|otte|ni|ti)\b', rewrite=r'xnumber_multiple', replace_global=True)
    # new_str = tf.strings.regex_replace(new_str, pattern=r'\b(?:anden|tredje|fjerde|femte|sjette|syvende|ottende|niende|tiende)(?:-)?', rewrite=r'xnumber_multiple', replace_global=True)


def remove_dash_regex():
    return '-'

def split_by_specials_regex():
    return '([^a-zæøåñäöîçíãúéïèüáëó0-9\s])'

def year_regex():
    return "(?:18|19|20)\d\d"

def digit_regex():
    return "\d+"

def countries_regex():
    countries_joined = ("(?:" + "|".join(countries) + ")")
    return "(?:^|\s)(?:nord|syd|øst|vest)?" + countries_joined + "s?\\b"

def nationalities_regex():
    nationalities_joined = ("(?:" + "|".join(nationalities) + ")")
    return "(?:^|\s)(?:nord|syd|øst|vest)?" + nationalities_joined + "(?:eren|erne|ere|ne|en|er|r|e|isk)?" + "(\w*)"

def weekday_regex():
    weekday_joined = ("(?:" + "|".join(weekdays) + ")")
    return "\\b" + weekday_joined + "(?:s|en)?\\b"

def tournament_regex():
    tournament_joined = ("(?:" + "|".join(tournaments) + ")")
    return "\\b" + tournament_joined + "(?: turnering|-turnering)?(?:s|et|er|en|ens|erne)?\\b"

def finals_regex():
    return "\w*finale"


regex_dict = {
    "remove_dash" : { "regex" : remove_dash_regex(), "replacewith" : r' '},
    "split_by_specials" : {  "regex" : split_by_specials_regex(), "replacewith" : r' \1 '},
    "countries" : {  "regex" : countries_regex(), "replacewith" :  ' xland' },
    "nationalities" :  {  "regex" : nationalities_regex(), "replacewith" : r' xnationality \1' },
    "weekdays" : {  "regex" : weekday_regex(), "replacewith" : "xweekday" },
    "tournaments" : {  "regex" : tournament_regex(), "replacewith" : "xtournament" },
    "finals" : {  "regex" : finals_regex(), "replacewith" : "xfinale" },
    "year" :  {  "regex" : year_regex(), "replacewith" : "xyear" },
    "digit" : {  "regex" : digit_regex(), "replacewith" : "xnumber" }
}



print(regex_dict["remove_dash"])


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
    x = tf.strings.lower(input_data, encoding='utf-8')
    x = tf.strings.regex_replace(x, pattern=regex_dict["remove_dash"]["regex"], rewrite=regex_dict["remove_dash"]["replacewith"])
    # x = split_included_specials(x)
    x = tf.strings.regex_replace(x, pattern=regex_dict["split_by_specials"]["regex"], rewrite=regex_dict["split_by_specials"]["replacewith"])
    # x = replace_tournament(tournaments)(x)
    x = tf.strings.regex_replace(x, pattern=regex_dict["tournaments"]["regex"], rewrite=regex_dict["tournaments"]["replacewith"])
    x = tf.strings.regex_replace(x, pattern=regex_dict["countries"]["regex"], rewrite=regex_dict["countries"]["replacewith"])
    # x = replace_weekday(weekdays)(x) 
    x = tf.strings.regex_replace(x, pattern=regex_dict["weekdays"]["regex"], rewrite=regex_dict["weekdays"]["replacewith"])
    # x = replace_finals(x)
    x =  tf.strings.regex_replace(x, pattern=regex_dict["finals"]["regex"], rewrite=regex_dict["finals"]["replacewith"])
    # x = replace_nationality(nationalities)(x)
    x =  tf.strings.regex_replace(x, pattern=regex_dict["nationalities"]["regex"], rewrite=regex_dict["nationalities"]["replacewith"])
    x =  tf.strings.regex_replace(x, pattern=regex_dict["year"]["regex"], rewrite=regex_dict["year"]["replacewith"])
    x =  tf.strings.regex_replace(x, pattern=regex_dict["digit"]["regex"], rewrite=regex_dict["digit"]["replacewith"])
    # x = replace_digits(x)
    return x

@tf.keras.utils.register_keras_serializable()
def vectorize_layer(max_features, sequence_length, standardization):
    return TextVectorization(
    standardize=standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length)

