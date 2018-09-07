import nltk
from train_data import training_data_generator
import urllib.request
from shifter import shifted_data_generator
from mappings import get_all_mappings, encode
from urllib.error import HTTPError
nltk.download('gutenberg')

def gutenberg_book_generator_from_website(start=1070, end=57482):
    for i in range(start, end):
        print(f'Using book {i}')
        try:
            with urllib.request.urlopen(f'http://www.gutenberg.org/files/{i}/{i}.txt') as response:
                with open('./current_book', 'w') as f:
                    f.write(str(i))
                yield response.read().decode('utf-8')
        except HTTPError as err:
            print(f'Unable to fetch book number {i}. error:', err)
            with open('./gutenberg_errors.txt', 'a') as f:
                f.write(f'{i}\n')
        except UnicodeDecodeError as err:
            print(f'Unable to decode unicode in book number {i}. error:', err)
            with open('./gutenberg_errors.txt', 'a') as f:
                f.write(f'{i}\n')
        except:
            print(f'unknown error on book {i}')
        
def gutenberg_book_generator():
    for id in nltk.corpus.gutenberg.fileids():
        yield nltk.corpus.gutenberg.raw(id)

def gutenberg_training_data_generator(sentence_length, batch_size, shift=True):
    for data in training_data_generator(gutenberg_book_generator, sentence_length, batch_size, shift):
        yield data




