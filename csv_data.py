import pandas as pd
import numpy as np
from collections import OrderedDict
import re
import collections
from create_vocab import remove_duplicates
from utils import replace_empty_space_variants
from utils import remove_trailing_space_and_full_stop
from utils import to_lower_case
from utils import convert_string_2_bool

def text_formatter(str_arr):
    xf = str_arr
    xf = [replace_empty_space_variants(text) for text in xf]
    xf = [remove_trailing_space_and_full_stop(text) for text in xf]
    xf = [x for x in xf if x!= '']
    xf = [to_lower_case(text) for text in xf]
    return xf

def get_all_articles(shuffle: bool):
    recent = get_recent_articles()
    archived = get_archived_articles()

    df = pd.concat([
        recent,
        archived
    ])

    # shuffle
    if shuffle == True:
        df = df.sample(frac=1).reset_index(drop=True)

    train_text = df.iloc[:, [0,1,2]].apply(' . '.join, axis=1).to_numpy()
    formatted_train_text = text_formatter(train_text)
    labels = df['isResult'].apply(lambda x: convert_string_2_bool(x)).to_numpy().astype(int)

    return formatted_train_text, labels


def get_recent_articles():

    import glob
    file_paths = glob.glob('articles_recent/*.csv')
    pds = []
    for f in file_paths:
        pds.append(pd.read_csv(f, encoding = "ISO-8859-1"))
    
    return pd.concat(pds)
    
def get_archived_articles():


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
    df_vm_fodbold_2 =  pd.read_csv('articles/vm_fodbold_2.csv', encoding = "ISO-8859-1")
    df_vm_fodbold_3 =  pd.read_csv('articles/vm_fodbold_3.csv', encoding = "ISO-8859-1")
    df_latest_new =  pd.read_csv('articles/latest_new.csv', encoding = "ISO-8859-1")
    df_bueskydning =  pd.read_csv('articles/bueskydning.csv', encoding = "ISO-8859-1")
    df_motorsport =  pd.read_csv('articles/motorsport.csv', encoding = "ISO-8859-1")
    df_dr_latest_2 =  pd.read_csv('articles/dr_latest_2.csv', encoding = "ISO-8859-1")
   
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
        df_vm_fodbold_2,
        df_vm_fodbold_3,
        df_latest_new,
        df_bueskydning,
        df_motorsport,
        df_dr_latest_2,
        ])
    
    return df


def column_data_2_list(columns, df):
    df_vocab_select_columns = df.iloc[:, columns]
    vocab_all_values = df_vocab_select_columns.values.ravel()
    return vocab_all_values

def vocab_2_dict(list_of_words):
    set_of_list = set(list_of_words)
    return OrderedDict.fromkeys(set_of_list)

def csv_list_to_list(csv_name):
    df = pd.read_csv(csv_name, encoding = "ISO-8859-1", header=None)
    df_list = df.fillna('').iloc[:,:].values.ravel().tolist()
    df_formatted_list = text_formatter(df_list)
    df_formatted_list_set = remove_duplicates(df_formatted_list)
    return df_formatted_list_set

def get_vocab_dict():
    df_ddo_vocab = pd.read_table('resources/ddo_fullforms_2020-08-26.csv', header=None)
    df_sport_lingo = pd.read_table('resources/sport_lingo.csv', encoding = "ISO-8859-1", header=None)

    names_to_omit = text_formatter(csv_list_to_list('resources/names_to_omit.csv'))

    ddo_vocab_set = column_data_2_list([0,1], df_ddo_vocab)
    sport_lingo_set = column_data_2_list([0], df_sport_lingo)

    vocab_arr = text_formatter(np.concatenate([ddo_vocab_set, sport_lingo_set]))

    vocab_arr = np.setdiff1d(vocab_arr, names_to_omit)

    vocab_dict = vocab_2_dict(vocab_arr)

    return vocab_dict

def check_duplicates(data_arr):
    duplicates = [item for item, count in collections.Counter(data_arr).items() if count > 1]
    for d in duplicates:
        print(d)
    assert(len(duplicates) == 0)

def split_data(data, percentage):
    train, labels = data

    l = len(train)
    p = l - int((percentage/100) * l)

    return (train[0:p], train[p:], labels[0:p], labels[p:])
