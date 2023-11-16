# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import OrderedDict


# %%
from csv_data import get_vocab_dict
from csv_data import get_all_articles
from csv_data import csv_list_to_list
from csv_data import split_data
from csv_data import check_duplicates


# # TODO : fjern ord der er kategorisert som "egennavn" i ddo_fullforms_2020-08-26.csv

ordered_dict = get_vocab_dict()
data_total, labels_total = get_all_articles(shuffle=True)
nationalities = csv_list_to_list('resources/nationalities.csv')
countries = csv_list_to_list('resources/countries.csv')
ignore_list = csv_list_to_list('resources/ignore_list.csv')

check_duplicates(data_total)

train_data, val_data, train_labels, val_labels = split_data((data_total, labels_total), 5)

print("Train data/labels length: ", len(train_data), len(train_labels))
print("Validation data/labels length: ", len(val_data),  len(val_labels))


# %%
import re

for i,tt in enumerate(data_total):
    if re.search(r'\bmålfarlig', tt, re.IGNORECASE):
        # if (labels_total[i] == 0):
            print(tt)
            print(labels_total[i])
        # print(train)



# %%
counts = np.bincount(train_labels)
# print(
#     "Number of positive samples in training data: {} ({:.2f}% of total)".format(
#         counts[1], 100 * float(counts[1]) / len(train_labels)
#     )
# )

print(counts)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]

class_weight = {0: weight_for_0, 1: weight_for_1}

# %%

# def combine_articles_to_csv():
#     df_sport_combined = df_sport.copy().drop('Link', axis=1)
#     df_sport_combined.to_csv('articles_temp/combined.csv')

# combine_articles_to_csv()


# %%
# duplicate_rows = df_sport.duplicated()
# print("Duplicates in data points: ")
# print(df_sport[duplicate_rows])

# %%
import time
isin_dict = False
def test_lookup_performance():
    word_to_check = "rune"
    start_time = time.time()

    for x in range(1000000):
        isin_dict = word_to_check in ordered_dict

    end_time = time.time()  
    assert(end_time - start_time < 1)
    print(isin_dict)

test_lookup_performance()

# isin_dict


# %%
from create_vocab import remove_duplicates
from create_vocab import remove_nationalities
from create_vocab import remove_non_frequent
from utils import remove_words_containing_digits
from create_vocab import remove_non_dict_words
from vectorization import to_lower
from vectorization import remove_dash
from vectorization import split_included_specials
from vectorization import standardize

def train_text_to_formatted_words(sentences):

    formatters = [
        to_lower, 
        remove_dash, 
        split_included_specials, 
    ]

    vocab_formatter_pipe = standardize(formatters)

    words_formatted = []
    for sentence in sentences:
        words = vocab_formatter_pipe(sentence).numpy().decode().split()
        words_formatted.extend(words)

    return words_formatted

words_formatted = train_text_to_formatted_words(train_data)

sentence_words_frequent = remove_non_frequent(words_formatted, 1)
words_arr_unique = remove_duplicates(sentence_words_frequent)
words_arr_unique = remove_words_containing_digits(words_arr_unique)
    
words_train_vocab, words_sport_lingo = remove_non_dict_words(words_arr_unique, ordered_dict)


# # TODO : brug tensorflow Tokenezier til at omdanne ord til tokens
# # TODO : søg i alle leksikoner, søg med og uden bindestreg
# # TODO : håndter tal ikke i ordbøger eks ( x-x eller x-årig)
# # TODO : lemmatizer : udelad bøjninger af samme navneord. eks : verdensmester/verdensmesteren
# # TODO : evt. grupper ord der ofte hænger sammen med nltk BigramFinder. eks vandt over
# TODO : fjern evt. også alle navne (fornavne og efternavne)  

print("total sports lingo words:", len(words_sport_lingo) )
print("total vocab:", len(words_train_vocab))



# %%

"håndbold" in words_train_vocab

# %%
file = open('words_sport_lingo.txt','w')
for item in words_sport_lingo:
	file.write(item+"\n")
file.close()

file = open('words_train_vocab.txt','w')
for item in sorted(words_train_vocab):
	file.write(item+"\n")
file.close()


# %%


for f in remove_duplicates(words_formatted):
    if f in words_sport_lingo and f not in ignore_list and len(f) > 1:
        print(f)


# %%


# TODO : lav en negativ liste også
# display most frequent words found in lingo words
# words_that_appear_once = []
# for f in frequent_words:
#     if f[0] < 2 and f[1] not in navne:
#         words_that_appear_once.append(f[1])

# # print(len(words_that_appear_once))
# sorted(words_that_appear_once)


# %%


# f = frequent_words[-300:]
# ff = []
# for w in frequent_words:
#     if w[1] in words_train_vocab and w[0] < 2:
#         ff.append(w[1])

# print(len(ff))
# ff
    

# %%


from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from vectorization import vect_layer_2_text
from vectorization import vectorize_layer
from vectorization import standardize
from vectorization import custom_standardization
from static_data import tournaments
from static_data import weekdays
from static_data import non_alpha
from static_data import word_generalization


# TODO : evt indikere hvilke navneord der starte med stort bogstav(egenavne), evt. lave et opslag for at undersøge ordklasse for det første ord i sætningen 
# TODO : test hvilke standarization funktioner giver bedre resultater 

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

# vectorization_pipe = standardize(vectorization_formatters)

words_train_vocab.extend(word_generalization)
words_train_vocab.extend(non_alpha)

# Model constants.
max_features = 7300
sequence_length = 60

vectorize = vectorize_layer(max_features, sequence_length, custom_standardization)

vectorize.adapt(words_train_vocab)
vectorization_vocab = vectorize.get_vocabulary()

print("Total vocab/max_features : ",  len(vectorization_vocab))

print (vect_layer_2_text(vectorize(["danmark skal med i OL, hvor danskere skal besejre svenskere"]), vectorization_vocab))

vectorize(["danmark skal med i OL, hvor danskere skal besejre svenskere"])


# %%
import json

from vectorization import regex_dict

with open('vocab.json', 'w',  encoding='utf8') as file:
    json.dump(vectorization_vocab, file, ensure_ascii=False)

with open('regexes.json', 'w',  encoding='utf8') as file:
    json.dump(regex_dict, file, ensure_ascii=False)



# %%
# for t in train_data[0:50]:
#     print("Original \n:", t)
#     print("Text from vectorized: \n", vect_layer_2_text(
#         vectorized_layer([t]), vect_vocab
#         ))
#     print("\n")
"håndbold" in vectorization_vocab

# %%
train_data_vect = vectorize(train_data)
val_data_vect = vectorize(val_data)


# print(type(train_data))

# train_data = np.array(train_data)
# val_data = np.array(val_data)

# print(type(n))


# %%
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



# %%
# from tensorflow.keras import layers
# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs

#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

# %%
m = tf.keras.metrics.BinaryAccuracy(threshold=0.97)
m.update_state([[1], [1], [0], [0]], [[0.98], [0.98], [0], [0]])
m.result().numpy()

# %%
from tensorflow.keras import layers
import random as python_random

def get_cnn_model():

    embedding_dim = 48

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    # x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    # x = layers.Conv1D(32, 11, padding="valid", activation="relu")(x)
    # x = layers.Conv1D(128, 9, padding="valid", activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(32, 7, padding="valid", activation="relu", strides=3)(x)
    # x = layers.Conv1D(32, 7, padding="valid", activation="relu", strides=3)(x)


    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # We add a vanilla hidden layer:
    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dropout(0.3)(x)


    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    
    cnn_model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return cnn_model



# %%
from tensorflow.keras import layers
import random as python_random


def get_transformer_model():

    embed_dim =  192 # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 512  # Hidden layer size in feed forward network inside transformer


    # A integer input for vocab indices.
    # inputs = tf.keras.layers.Input(dtype=tf.string, shape=(1,))
    inputs = tf.keras.Input(shape=(sequence_length,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    # x = layers.Embedding(max_features, embed_dim)(inputs)

    # text_model_catprocess2 = vectorize(inputs)

    embedding_layer = TokenAndPositionEmbedding(sequence_length, max_features, embed_dim)
    x = embedding_layer(inputs)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dropout(0.2)(x)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    # x = layers.Dropout(0.1)(x)

    # x = transformer_block(x)



    # Conv1D + global max pooling
    # x = layers.Conv1D(128, 10, padding="valid", activation="relu", strides=3)(x)
    # x = layers.Conv1D(128, 10, padding="valid", activation="relu", strides=3)(x)


    x = layers.GlobalMaxPooling1D()(x)
    # x = layers.Dropout(0.5)(x)

    # We add a vanilla hidden layer:
    # x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    # x = layers.Dropout(0.5)(x)



    transformer_model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    bn =   tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.97)
    
    
    transformer_model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    return transformer_model


# %%

def prepare_model(name):
    if (name == "cnn"):
       return get_cnn_model()
    elif (name == "transformer"):
       return get_transformer_model()
  

def filter_max_accuracy(history, threshold = 0.95):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    list = []
    for x in range(len(acc)):
        if (acc[x] > threshold):
            list.append(val_acc[x])

    return np.array(list)

models = ["cnn", "transformer"]

callback_3_loss = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)


def mean_model_accuracy(mode_names, iterations, epochs = 20):

  
    results = []

    for name in range(len(mode_names)):
        model_name = mode_names[name]
        val_accuracies = []
        
        for x in range(iterations):
            model = prepare_model(model_name)

            # Fit the model using the train and test datasets.
            history = model.fit(train_data_vect, train_labels, epochs=epochs, batch_size=6, validation_data=(val_data_vect, val_labels), callbacks=[callback_3_loss])

            max_val_acc = filter_max_accuracy(history)
            val_accuracies.append(max(max_val_acc))
            print(max(max_val_acc))
            print(val_accuracies)
        
        d = dict(name = model_name, results = np.mean(np.squeeze(np.array(val_accuracies))))
        results.append(d)
        
    return results


# %%
# mean_results = mean_model_accuracy(models, 8)
# mean_results

# %%
def result_format_round(result):
    return round(result)

def result_format_none(result):
    return result

def print_model_score(model):
    score = model.evaluate(val_data_vect, val_labels, verbose=0)
    print("Validation loss:", score[0])
    print("Validations accuracy:", score[1])

def print_validation_results(predictions, val_data, labels, formatter, only_incorrect = False):
    print("Number of predictions", len(predictions))
    n_correct = 0
    for x in range(len(val_data)):
        correct_prediction = result_format_round(labels[x]) == result_format_round(predictions[x][0])
        if correct_prediction:
            n_correct += 1

        # if correct_prediction == False and labels[x] == 0:
        print("VALIDATION SAMPLE TEXT: \n" ,val_data[x])
        print("VALIDATION SAMPLE DE-VECTORIZED: \n" ,vect_layer_2_text(val_data_vect[x], vectorization_vocab))
        print("LABEL -- :" , labels[x])
        print("PREDICTION -- :" , formatter(predictions[x][0]), " ---- float: ", predictions[x][0])
        print("CORRECT PREDICTION: ", correct_prediction)
        print("\n")

    print("Number correct: ", n_correct)

# %%

epochs= 6
transformer_model = get_transformer_model()

transformer_history = transformer_model.fit(train_data_vect, train_labels, epochs=epochs, batch_size=120, validation_data=(val_data_vect, val_labels),  class_weight=class_weight,)


# %%
transformer_model.summary()

# %%

# epochs= 7
# cnn_model = get_cnn_model()

# transformer_history = cnn_model.fit(train_data_vect, train_labels, epochs=epochs, batch_size=40, validation_data=(val_data_vect, val_labels), class_weight=class_weight)

# %%
# cnn_model.summary()

# %%
def print_results(model):
    np.set_printoptions(precision = 5, suppress = True)
    predictions = model.predict(val_data_vect)
    print_model_score(model)
    print("\n")
    print_validation_results(predictions, val_data, val_labels, result_format_round)
  

# %%
predictions = transformer_model.predict(val_data_vect[0:1])
val_data_vect[0]

len(predictions)

predictions

# %%
print("--- TRANSFORMER ---")
print_results(transformer_model)

# %%

# print("--- CNN ---")
# print_results(cnn_model)

# %%
import os

# Set up a logs directory, so Tensorboard knows where to look for files.

ll = transformer_model.layers[1]
ll_weights = ll.get_weights()[0]

# print(ll_weights.shape)
ll_weights


# %%
##import I/O module in python
import io

##open the text stream for vectors
vectors = io.open('vectors.tsv', 'w', encoding='utf-8')

##open the text stream for metadata
meta = io.open('meta.tsv', 'w', encoding='utf-8')


##write each word and its corresponding embedding
for index in range(1, len(vectorization_vocab)):
  word = vectorization_vocab[index]  # flipping the key-value in word_index
  embeddings = ll_weights[index]
  meta.write(word + "\n")
  vectors.write('\t'.join([str(x) for x in embeddings]) + "\n")

##close the stream
vectors.close()
meta.close()

# %%
# from nltk import collocations
# bigram_measures = collocations.BigramAssocMeasures()
# finder = collocations.BigramCollocationFinder.from_words(["New", "York", "is", "big", "New", "York", "is", "dirty"])
# finder.ngram_fd.items()



# %%
# import lemmy
# # Create an instance of the standalone lemmatizer.
# lemmatizer = lemmy.load("da")

# # Find lemma for the word 'akvariernes'. First argument is an empty POS tag.
# lemmatizer.lemmatize("NOUN", "verdensetter")



# %%
# import nltk as nltk
# # from string import punctuation
# # from nltk.corpus import stopwords
# # nltk.download('stopwords')

# # da_stopwords = stopwords.words("danish")


# %%
# A string input
inputs = tf.keras.Input(shape=(1,), dtype="string")
# Turn strings into vocab indices

indices = vectorize(inputs)
# Turn vocab indices into predictions
outputs = transformer_model(indices)

# Our end to end model
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)


# %%
print("\nResults:")


print(end_to_end_model.predict(
    [
      "Fodbold . Fjerritslev vinder over Vordingborg. Træner kommenterer på historisk kamp",
       "SPORT . Hun vandt bronze i mandags Roer Anne Dsane Andersen har som 24-årig vundet bronze ved OL",
       "Badminton . Axelsen frustreret over nederlag. Viktor Axelsen trænger til ferie efter nedturen",
      "OL . Det blev til en flot medalje til Malene dfhsds. 'Jeg er meget lykkelig for resultatet'",
      "Badminton . Dansker er videre til finalerne. dsfsdf sfdsdf bankede Fdfsdf fra Kina og skal spille i finalen på onsdag",
     ]))


print("\n NON-Results:") 
print(end_to_end_model.predict(
    [
      "Badminton . Dansker er videre til finalerne. dsfsdf sfdsdf skal spille i finalen på onsdag",
      "OL Meget skal ske før en medalje kommer inden for rækkevidde. Dressurrytter Malene dsds har mistet troen på success",
      "Fodbold . Træner for Fjerritslev ser frem til sejr over Vordingborg. 'Det bliver en historisk kamp'",
      "Fodbold . De danske spillere skal op imod Sverige, som de tabte til i 2022",
      "Fodbold . De danske spillere vil forsøge at besejre Tyrkiet den kommende Lørdag i VM-kamp. Tyrkiet har aldriv været i en VM-finale",
      "Fodbold . De danske spillere tror på sejr mod Tyrkiet. 'Den skal vindes'",
      "Skisport . Sverige drømmer om flere medaljer og sejre til næste års OL . Træner forventer flere gode resultater",
      "Boksning . Kesler vil overraske alle og gøre det umulige. 'Jeg vinder i VM'",
      "Boksning . Kesler med stor selvtillid: 'Det bliver guld eller sølv til VM'"
     ]))

# %%
print("\n IN-BETWEEN-Results:") 
print(end_to_end_model.predict(
    [
       "Fodbold . Fjerritslev vandt i lørdags over Vordingborg 1-0. Den danske anfører dasdad dasdasd triumferer",
       "Fodbold . Fjerritslev vandt i lørdags over Vordingborg. Den danske anfører adasdasdd daddas triumferer",
       "Fodbold . Fjerritslev vandt i lørdags over Vordingborg. Efter kampen meddelyte den danske anfører sdfd sdfdf, at han skal under kniven",
       "Fodbold . Superliga-profil efter storsejr over Vordingborg. ' Den danske anfører fsdsdff sdffsd skal opereres og er ude i flere måneder",
       "Fodbold . Superliga-profil har meddelelse efter sejr. Den danske anfører fdfd sdffdf skal opereres og er ude i flere måneder",
       "Fodbold . Superliga-profil kan se frem til en længere pause. Den danske anfører fdfd sdfff skal opereres og er ude i flere måneder",
       
        ]))

# %%
transformer_model.save('transformer_model')
loaded_model = keras.models.load_model('transformer_model')

# %%
# import tensorflowjs as tfjs

# tfjs

# %%
# !tensorflowjs_converter --input_format tf_saved_model "model" ./jsmodel

# %%
# loaded_model.predict(
#     [
#        "Fodbold . Fjerritslev vandt i lørdags over Vordingborg 1-0. Den danske anfører dasdad dasdasd triumferer",
#        "Fodbold . Fjerritslev vandt i lørdags over Vordingborg. Den danske anfører adasdasdd daddas triumferer",
#        "Fodbold . Fjerritslev vandt i lørdags over Vordingborg. Efter kampen meddelyte den danske anfører sdfd sdfdf, at han skal under kniven",
#        "Fodbold . Superliga-profil efter storsejr over Vordingborg. ' Den danske anfører fsdsdff sdffsd skal opereres og er ude i flere måneder",
#        "Fodbold . Superliga-profil har meddelelse efter sejr. Den danske anfører fdfd sdffdf skal opereres og er ude i flere måneder",
#        "Fodbold . Superliga-profil kan se frem til en længere pause. Den danske anfører fdfd sdfff skal opereres og er ude i flere måneder",
       
#         ])

# %%
# !pip install tensorflowjs
# !tensorflowjs_converter --input_format keras "cnn_emnist.h5" ./jsmodel

# %%
# import tensorflowjs as tfjs

# docker run -it --rm tensorflow/tensorflow:latest-gpu-jupyter \
#    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"


