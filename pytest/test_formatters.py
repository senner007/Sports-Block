from create_vocab import split_sentences
import numpy as np

# Arrange
sentence = ["1.division em-hold gog's 25"] 
expected = np.array(["1", ".", "division", "em", "-", "hold", "gog", "'" , "s", "25"])
# Act
actual = split_sentences(sentence)
# Assert
assert(np.array_equal(expected, actual))