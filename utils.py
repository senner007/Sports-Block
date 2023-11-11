import re

def to_lower_case(str_param):
    return str_param.lower()
    
def replace_empty_space_variants(text):
    # replace \x85, '\r', '\n', \xa0
    pattern = r'[\s\xa0]'
    return re.sub(pattern, ' ', text)

def convert_string_2_bool(x):
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

def remove_words_containing_digits(words):
    return [x for x in words if any(char.isdigit() for char in x) == False]

def remove_trailing_space_and_full_stop_arr(strs):
    stripped = []
    for s in  strs:
        stripped.append(s.rstrip(". ").strip()) # removes trailing newline and full stop and space
    return stripped

def remove_trailing_space_and_full_stop(str_param):
    return str_param.rstrip(". ").strip() # removes trailing newline and full stop and space
