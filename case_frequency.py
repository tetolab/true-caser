# Case weight generator


import string

lowercase = list(string.ascii_lowercase)
uppercase = list(string.ascii_uppercase)

# Frequency Data Source:
# Case-sensitive letter and bigram frequency counts
# from large-scale English corpora
# Authors: MICHAEL N. JONES and D. J. K. MEWHORT
case_amount_upper = {
    'A': 280_937,
    'B': 169_474,
    'C': 229_363,
    'D': 129_632,
    'E': 138_443,
    'F': 100_751,
    'G': 93_212,
    'H': 123_632,
    'I': 223_312,
    'J': 78_706,
    'K': 46_580,
    'L': 106_984,
    'M': 259_474,
    'N': 205_409,
    'O': 105_700,
    'P': 144_239,
    'Q': 11_659,
    'R': 146_448,
    'S': 304_971,
    'T': 325_462,
    'U': 57_488,
    'V': 31_053,
    'W': 107_195,
    'X': 7_578,
    'Y': 94_297,
    'Z': 5_610
}

case_amount_lower = {
    'a': 5_263_779,
    'b': 866_156,
    'c': 1_960_412,
    'd': 2_369_820,
    'e': 7_741_842,
    'f': 1_296_925,
    'g': 1_206_747,
    'h': 2_955_858,
    'i': 4_527_332,
    'j': 65_856,
    'k': 460_788,
    'l': 2_553_152,
    'm': 1_467_376,
    'n': 4_535_545,
    'o': 4_729_266,
    'p': 1_255_579,
    'q': 54_221,
    'r': 4_137_949,
    's': 4_186_210,
    't': 5_507_692,
    'u': 1_613_323,
    'v': 653_370,
    'w': 1_015_656,
    'x': 123_577,
    'y': 1_062_040,
    'z': 66_423
}

case_amounts = {**case_amount_lower, **case_amount_upper}


def generate_ratios():
    ratios = {}
    for letter, amount in case_amounts.items():
        if letter.isupper():
            lower_amount = case_amounts.get(letter.lower())
            ratio = amount / (amount + lower_amount)
            ratios[letter] = ratio
        else:
            upper_amount = case_amounts.get(letter.upper())
            ratio = amount / (amount + upper_amount)
            ratios[letter] = ratio
    return ratios


def generate_weights():
    return {k: 1 - v for k, v in generate_ratios().items()}
