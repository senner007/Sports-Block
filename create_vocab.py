import re

def split_specials(word):
    words_new = []
    parts = re.findall(r"[A-ZÆØÅa-zæøå0-9]+|\S", word)
    words_new.extend([x for x in parts])
    return words_new

# def contains_non_alphanumeric(word):
#     return bool(re.search(r'[^a-zæøåA-ZÆØÅ0-9]', word))


def split_sentences(sentences):
    words_arr = []
    for ind, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            w = split_specials(word)
            words_arr.extend([x.lower() for x in w])
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

def remove_danske_navne(words, danske_navne):
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

def add_non_alpha_numeric(words, non_alpha):
    ws = words
    ws.extend(non_alpha)
    return ws
