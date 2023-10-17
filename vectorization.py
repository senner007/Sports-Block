from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import numpy as np
from static_data import non_alpha

def increment(x):
    return x + 1

# TODO : evt indikere hvilke navneord der starte med stort bogstav(egenavne), evt. lave et opslag for at undersøge ordklasse for det første ord i sætningen 


def to_lower(word):
    return tf.strings.lower(word, encoding='utf-8')

def split_dash(word):
  return tf.strings.regex_replace(word, '-', ' ')

def split_specials(word):

    new_str = word
    for sign in non_alpha:
        r = "\\" + sign
        new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite=" " + sign + " ")

    return new_str

def replace_digits(word):
    return tf.strings.regex_replace(word, pattern=r'\d+', rewrite=r'xx')


def replace_countries(countries):
    def replace_countries(word):

        new_str = word
        for sign in countries:
            r = sign + "s?\\b"
            new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="x_land")

        return new_str
    return replace_countries


def replace_tournament(tournaments):
    def replace_tournament(word):

        new_str = word
        for sign in tournaments:
            r = "\\b" + sign + "\\w*(?:(?:(\\sturnering)|-turnering)\\w*)?" + "\\b"

            new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="x_tournament")

        return new_str
    return replace_tournament

def replace_nationality(nationalities):
    def replace_nationality(word):
        new_str = word
        for sign in nationalities:
            r = "\\b" + sign + "\\w*"
            new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="x_nationality")

        return new_str
    return replace_nationality

def replace_weekday(weekdays):
        def replace_weekday(word):
            new_str = word
            for sign in weekdays:
                r = "\\b" + sign + "\\w*"
                new_str = tf.strings.regex_replace(new_str, pattern=r, rewrite="x_weekday")
            return new_str
        return replace_weekday



# TODO : test hvilke standarization funktioner giver bedre resultater 

# def custom_standardization(x):
#     x = to_lower(x)
#     x = split_dash(x)
#     x = split_specials(x)
#     x = replace_tournament(x)
#     x = replace_countries(x)
#     x = replace_weekday(x)    
#     # x = replace_nationality(x)                     
#     return replace_digits(x)

def standardize(func_arr):
    def iterate_funcs(x):
        val = x
        for f in func_arr:
            val = f(val)
        return val
    return iterate_funcs


# arrs = [
#     to_lower, split_dash, split_specials, replace_tournament, replace_countries, replace_countries, replace_weekday, replace_digits
# ]

# s = standardize(arrs)

def vect_layer_2_text(vect, vect_vocab):
    return np.array([vect_vocab[x] for x in np.trim_zeros(np.squeeze(vect.numpy()))])

def vectorize_layer(max_features, sequence_length, standardization):
    return TextVectorization(
    standardize=standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length)



# # Model constants.
# max_features = 5600
# sequence_length = 100

# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=max_features,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )

# add the word 'xx' to the allowed vocabulary representing all numbers
# word_generalization = [ "xx", "x_land", "x-tournament", "x_nationality", "x_weekday"]
# word_generalization.extend(tournaments.values())

# def prepare_vocab(words):
#     words_copy = words.copy()
#     words_copy.extend(word_generalization)
#     return words_copy

# text_ds = vectorize_layer.adapt(prepare_vocab(words_train_vocab))
# vect_vocab = vectorize_layer.get_vocabulary()

