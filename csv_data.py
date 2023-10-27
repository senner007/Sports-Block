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
    df_fodbold_tv2 = pd.read_csv('articles/fodbold_tv2.csv', encoding = "ISO-8859-1")
    df_fodbold_2_tv2 = pd.read_csv('articles/fodbold_tv2_2.csv', encoding = "ISO-8859-1")
    df_formel_1_tv2 = pd.read_csv('articles/formel_1_tv2.csv', encoding = "ISO-8859-1")
    df_haandbold_tv2 = pd.read_csv('articles/haandbold.csv', encoding = "ISO-8859-1")
    df_skisport_tv2 = pd.read_csv('articles/skisport.csv', encoding = "ISO-8859-1")
    df_basketball_tv2 = pd.read_csv('articles/basketball.csv', encoding = "ISO-8859-1")
    df_ol = pd.read_csv('articles/ol.csv', encoding = "ISO-8859-1")
    df_fodbold_3 = pd.read_csv('articles/fodbold_3.csv', encoding = "ISO-8859-1")
    df_tennis = pd.read_csv('articles/tennis.csv', encoding = "ISO-8859-1")
    df_atletik = pd.read_csv('articles/atletik.csv', encoding = "ISO-8859-1")
    df_sejlsport = pd.read_csv('articles/sejlsport.csv', encoding = "ISO-8859-1")
    df_boksning = pd.read_csv('articles/boksning.csv', encoding = "ISO-8859-1")
    df_tour_de_france = pd.read_csv('articles/tour_de_france.csv', encoding = "ISO-8859-1")
    df_vinterol = pd.read_csv('articles/vinterol.csv', encoding = "ISO-8859-1")
    df_esport =  pd.read_csv('articles/esport.csv', encoding = "ISO-8859-1")
    df_vm_fodbold_1 =  pd.read_csv('articles/vm_fodbold_1.csv', encoding = "ISO-8859-1")
    df_vm_fodbold_3 =  pd.read_csv('articles/vm_fodbold_3.csv', encoding = "ISO-8859-1")
    df_latest_new =  pd.read_csv('articles/latest_new.csv', encoding = "ISO-8859-1")
    df_bueskydning =  pd.read_csv('articles/bueskydning.csv', encoding = "ISO-8859-1")



    df = pd.concat([
        df_sport_latest, 
        df_sport_latest_tv2, 
        df_sport_2019, 
        df_sport_2020, 
        df_sport_2022, 
        df_sport_politiken, 
        df_ishockey_tv2, 
        df_badminton_tv2, 
        df_nfl_tv2,
        df_cykling_tv2, 
        df_fodbold_tv2, 
        df_fodbold_2_tv2, 
        df_formel_1_tv2, 
        df_haandbold_tv2, 
        df_skisport_tv2, 
        df_basketball_tv2,
        df_ol,
        df_fodbold_3,
        df_tennis,
        df_atletik,
        df_sejlsport,
        df_boksning,
        df_tour_de_france,
        df_vinterol,
        df_esport,
        df_vm_fodbold_1,
        df_vm_fodbold_3,
        df_latest_new,
        df_bueskydning
        ])
    
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



    tt = list(train)
    ll = list(labels)

    count = 0
    i = 0
    val_data = []
    val_labels = []

    while count < ((l-p) /2):
        if ll[i] == 1:
            val_data.append(tt[i])
            val_labels.append(ll[i])
            del tt[i]
            del ll[i]
            count += 1

        i += 1  

    count = 0
    i = 0
    
    while count < ((l-p) /2):
        if ll[i] == 0:
            val_data.append(tt[i])
            val_labels.append(ll[i])
            del tt[i]
            del ll[i]
            count += 1

        i += 1   

    return (np.array(tt), np.array(val_data), np.array(ll), np.array(val_labels))

    # train, labels = data

    # l = len(train)
    # p = l - int((percentage/100) * l)
    # half_val_length = int((l -p) / 2)

    # # tt = list(train)
    # # ll = list(labels)

    # pos_ind = [i for (i,x) in enumerate(labels) if x == 0]
    # neg_ind = [i for (i,x) in enumerate(labels) if x == 1]

    # total_ind = pos_ind[:half_val_length] + neg_ind[:half_val_length]
    # print(total_ind)

    # val_data, val_labels = zip(*([(train[x], labels[x]) for x in total_ind]))
    # print(labels[:300])
    # train = np.delete(train, total_ind)
    # labels = np.delete(labels, total_ind)

    # return (train, np.array(val_data), labels, np.array(val_labels))




    # return (train[0:p], train[p:], labels[0:p], labels[p:])

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