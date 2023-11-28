import re
import numpy as np

def split_specials(word):
    return re.findall(r"[A-ZÆØÅa-zæøå0-9]+|\S", word)

def split_sentences(sentences):
    words_arr = []
    for sentence in sentences:
        words = split_specials(sentence)
        words_arr.extend(words)
    return words_arr

def remove_duplicates(words):
    return list(set(words))

def remove_nationalities(words, nationalities):
    words_minus_nationalities = []
    for w in words:
        result = any(w.startswith(item) for item in nationalities)
        if result == False:
            words_minus_nationalities.append(w)
    
    return list(set(words_minus_nationalities))

def remove_varioius_names(words, danske_navne):
    words_minus_danske_navne = []
    for w in words:
        result = any(w.startswith(item) for item in danske_navne)
        if result == False:
            words_minus_danske_navne.append(w)
    
    return list(set(words_minus_danske_navne))

def remove_danske_fornavne(words, danske_fornavne):
    print(danske_fornavne)
    words_minus_danske_navne = []
    for w in words:
        result = any(re.compile(fr"^{item}s?\b").search(w) for item in danske_fornavne) # TODO : test me!
        if result == False:
            words_minus_danske_navne.append(w)
    
    return list(set(words_minus_danske_navne))

def remove_non_dict_words(words, dict):
    words_in_dict = []
    words_not_in_dict = []
    for w in words:
        isin_dict = w in dict
        if isin_dict == True:
            words_in_dict.append(w)
        else:
            words_not_in_dict.append(w)
     
    return words_in_dict, words_not_in_dict

def remove_stopwords(words):
    words_minus_danske_navne = []
    for w in words: # TODO : test me!
        pattern = r'\b(?:et|en)\b'  # https://regex101.com/r/W9UHRq/1
        if not re.search(pattern, w, re.IGNORECASE):
            words_minus_danske_navne.append(w)
    
    return list(set(words_minus_danske_navne))


def words_by_frequency(arr):
    np_array = np.array(arr, dtype=object)
    unique, counts = np.unique(np_array, return_counts=True)
    aa = np.asarray((unique, counts)).T
    return np.flip(aa[aa[:, 1].argsort()])

def remove_non_frequent(words_arr, threshold):
    words_dict = words_by_frequency(words_arr)
    words_above_threshold = []
    for word_freq in words_dict:
        if word_freq[0] > threshold:
            words_above_threshold.append(word_freq[1])
    return words_above_threshold
