import tensorflow as tf
import pandas as pd
import numpy as np

print(tf.__version__)

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from vectorization import replace_finals
from vectorization import replace_nationality
from vectorization import to_lower
from vectorization import remove_dash
from vectorization import split_included_specials
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
from vectorization import custom_standardization



loaded_model = tf.keras.models.load_model('end_to_end_model')

loaded_model

print(loaded_model.predict(
    [
      "Fodbold . Fjerritslev vinder over Vordingborg. Træner kommenterer på historisk kamp",
       "SPORT . Hun vandt bronze i mandags Roer Anne Dsane Andersen har som 24-årig vundet bronze ved OL",
       "Badminton . Axelsen frustreret over nederlag. Viktor Axelsen trænger til ferie efter nedturen",
      "OL . Det blev til en flot medalje til Malene dfhsds. 'Jeg er meget lykkelig for resultatet'",
      "Badminton . Dansker er videre til finalerne. dsfsdf sfdsdf bankede Fdfsdf fra Kina og skal spille i finalen på onsdag",
     ]))