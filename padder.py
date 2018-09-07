def pad(sentence, max_length, padding_symbol=' '):
    if len(sentence) >= max_length:
        return sentence[:max_length]
    else:
        difference = max_length - len(sentence)
        sentence = list(sentence)
        sentence.extend([padding_symbol] * difference)
        return sentence
