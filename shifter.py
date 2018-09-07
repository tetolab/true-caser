from more_itertools import flatten

def generate_shifted_data(sentence, padding_symbol=0, max_len_of_sentence=32):
    padded_sentences = []
    sentence = list(flatten(sentence))
    length_of_sentence = len(sentence)
    difference = max_len_of_sentence - length_of_sentence
    
    if length_of_sentence < max_len_of_sentence:
        sentence.extend([padding_symbol] * difference)
        
    # right pad
    for i in range(length_of_sentence):
        if length_of_sentence - i < max_len_of_sentence: 
            new_sentence = sentence[i:length_of_sentence]
            new_sentence.extend([padding_symbol] * (max_len_of_sentence - length_of_sentence + i))
        else: 
            new_sentence = sentence[i:i+max_len_of_sentence]
        padded_sentences.append(new_sentence)
    # left pad
    for i in range(length_of_sentence):
        if length_of_sentence - i < max_len_of_sentence:    
            new_sentence = [padding_symbol] * (max_len_of_sentence - length_of_sentence + i)
            new_sentence.extend(sentence[0:(length_of_sentence - i)])
            padded_sentences.append(new_sentence)
    return padded_sentences

# lol
def shifted_data_generator(sentence, padding_symbol=0, max_len_of_sentence=32):
    sentence = list(flatten(sentence))
    length_of_sentence = len(sentence)
    difference = max_len_of_sentence - length_of_sentence
    
    if length_of_sentence < max_len_of_sentence:
        sentence.extend([padding_symbol] * difference)
        
    # right pad
    for i in range(length_of_sentence):
        if length_of_sentence - i < max_len_of_sentence: 
            new_sentence = sentence[i:length_of_sentence]
            new_sentence.extend([padding_symbol] * (max_len_of_sentence - length_of_sentence + i))
        else: 
            new_sentence = sentence[i:i+max_len_of_sentence]
        yield new_sentence
    # left pad
    for i in range(length_of_sentence):
        if length_of_sentence - i < max_len_of_sentence:    
            new_sentence = [padding_symbol] * (max_len_of_sentence - length_of_sentence + i)
            new_sentence.extend(sentence[0:(length_of_sentence - i)])
            yield new_sentence