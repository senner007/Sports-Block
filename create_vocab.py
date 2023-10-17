import re

def split_specials(word):
    words_new = []
    parts = re.findall(r"[A-ZÆØÅa-zæøå0-9]+|\S", word)
    words_new.extend([x for x in parts])
    return words_new

def contains_non_alphanumeric(word):
    return bool(re.search(r'[^a-zæøåA-ZÆØÅ0-9]', word))

def remove_numeric(words):
    return [x for x in words if any(char.isdigit() for char in x) == False]
    
def split_sentences(sentences):
    words_arr = []
    for ind, sentence in enumerate(sentences):
        sentence_trimmed = sentence.strip()
        words = sentence_trimmed.split()
        for word in words:
            w = split_specials(word)
            words_arr.extend([x.lower() for x in w])
    return words_arr


def remove_duplicates(words):
    return list(set(words))

def remove_nationalities(words, nationalities):
    words_minus_nationalities = []
    for w in words:
        result = any(w.startswith(item.lower()) for item in nationalities)
        if result == False:
            words_minus_nationalities.append(w)
    
    return list(set(words_minus_nationalities))

def remove_danske_navne(words, danske_navne):
    words_minus_danske_navne = []
    for w in words:
        result = any(w.startswith(item.lower()) for item in danske_navne)
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

def add_non_alpha_numeric(words, non_alpha):
    ws = words
    ws.extend(non_alpha)
    return ws


# words_arr = split_sentences(train_data)
# words_arr_unique = remove_duplicates(words_arr)
# words_arr_unique = remove_nationalities(words_arr_unique, nationalities)
# words_arr_unique = remove_numeric(words_arr_unique)

# words_train_vocab, words_sport_lingo = remove_non_dict_words(words_arr_unique, ordered_dict)

# words_train_vocab = add_non_alpha_numeric(words_train_vocab, non_alpha)

# print(words_train_vocab)


