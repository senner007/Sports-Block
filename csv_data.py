import pandas as pd
import numpy as np
from collections import OrderedDict
import re

def format_2_bool(x):
    if type(x) == bool:
        return x
    assert(type(x) == str)
    x_copy = x
    x_copy = x_copy.strip()
    x_copy = x_copy.lower()
    assert(x_copy == "true" or x_copy == "false")
    if x_copy == "true":
        return True
    else:
        return False

def get_sports():
    df_sport_latest = pd.read_csv('articles/sports_articles.csv', encoding = "ISO-8859-1")
    df_sport_latest_tv2 = pd.read_csv('articles/sports_articles_tv2.csv', encoding = "ISO-8859-1")
    df_sport_2019 = pd.read_csv('articles/sports_articles_2019.csv', encoding = "ISO-8859-1")
    df_sport_2020 = pd.read_csv('articles/sports_articles_2020.csv', encoding = "ISO-8859-1")
    df_sport_2022 = pd.read_csv('articles/sports_articles_2022.csv', encoding = "ISO-8859-1")
    df_sport_politiken = pd.read_csv('articles/sports_articles_politiken.csv', encoding = "ISO-8859-1")
    df_sport_mixed = pd.read_csv('articles/mixed.csv', encoding = "ISO-8859-1")
    df_ishockey_tv2 = pd.read_csv('articles/ishockey-tv2.csv', encoding = "ISO-8859-1")
    df_badminton_tv2 = pd.read_csv('articles/badminton-tv2.csv', encoding = "ISO-8859-1")
    df_nfl_tv2 = pd.read_csv('articles/nfl_tv2.csv', encoding = "ISO-8859-1")
    df_cykling_tv2 = pd.read_csv('articles/cykling_tv2.csv', encoding = "ISO-8859-1")



    df = pd.concat([df_sport_latest, df_sport_latest_tv2, df_sport_2019, df_sport_2020, df_sport_2022, df_sport_politiken, df_ishockey_tv2, df_badminton_tv2, df_nfl_tv2,df_cykling_tv2])
    df = df.sample(frac=1).reset_index(drop=True)
    df['isResult'] = df['isResult'].apply(lambda x: format_2_bool(x))

    for d in df['isResult']:
        assert(isinstance(d, bool) == True)
        
    return df


def vocab_2_pdset(columns, df):
    df_vocab_select_columns = df.iloc[:, columns]
    vocab_all_values = df_vocab_select_columns.values.ravel()
    return set(vocab_all_values)

def vocab_2_dict(sets):
    assert(len(sets) == 2)
    word_set = sets[0].union(sets[1])
    df = pd.DataFrame(list(word_set), columns=["Words"])
    df.sort_values(by="Words", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return OrderedDict.fromkeys(word_set)

def csv_to_list(csv_name):
    df_nationalities = pd.read_csv(csv_name, encoding = "ISO-8859-1", header=None)
    nationalities = df_nationalities.fillna('').iloc[:,:].values.ravel().tolist()
    return [x.strip().lower() for x in nationalities if x!= '']

# TODO : fjern ord der er kategorisert som "egennavn" i ddo_fullforms_2020-08-26.csv

def get_vocab_dict():
    # df_ods_vocab = pd.read_table('ods_fullforms_2020-08-26.csv', header=None)
    df_ddo_vocab = pd.read_table('ddo_fullforms_2020-08-26.csv', header=None)
    # df_vocab = pd.read_table('cor1.02.tsv', header=None)
    df_sport_lingo = pd.read_table('sport_lingo.csv', header=None)

    # vocab_set = vocab_2_pdset([1,3], df_vocab)
    # ods_vocab_set = vocab_2_pdset([0,1], df_ods_vocab)
    ddo_vocab_set = vocab_2_pdset([0,1], df_ddo_vocab)
    sport_lingo_set = vocab_2_pdset([0], df_sport_lingo)

    d = vocab_2_dict([ddo_vocab_set, sport_lingo_set])
    d = {key.lower() if isinstance(key, str) else key: value for key, value in d.items()}

    return d

# U+00a0

def remove_hidden_spaces(text):
    pattern = r'[\s\u00A0]+'
    return re.sub(pattern, ' ', text)


def extract_data(df):
    train_text = df.iloc[:, [0,1,2]].apply(' . '.join, axis=1).replace('\xa0', ' ', regex=True).to_numpy()
    train_text = [remove_hidden_spaces(text) for text in train_text]

    labels = df['isResult'].to_numpy().astype(int)

    return train_text, labels

def split_data(data, percentage):
    train, labels = data
    l = len(train)
    p = l - int((percentage/100) * l)
    return (train[0:p], train[p:], labels[0:p], labels[p:])

# ordered_dict = get_vocab_dict()
# df_sport = get_sports()

# nationalities = get_csv('nat3.csv')
# countries = get_csv('countries.csv')
# navne = get_csv('navne.csv')

# train_data, val_data, train_labels, val_labels = split_data(extract_data(df_sport), 6)

# # train_data_results = get_results_in_data(train_data, train_labels)

# # print("Total data: ", len(train_text))
# print("Train data length: ", len(train_data), len(train_labels))
# print("Validation data length: ", len(val_data),  len(val_labels))