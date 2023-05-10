import math
import random
from augmentation_techniques.word2vec_rusvectores import get_neighbors_with_word2vec, get_neighbors_with_rusvectores
from gensim.models import Word2Vec, KeyedVectors

file = open("../data/stop_words_russian.txt", "r")
stop_words = file.read().splitlines()


def random_deletion(text: str, part: float = 0.3) \
        -> str:  # аугментированный текст

    words = text.split()

    length = len(words)
    length_20percent = math.trunc(length * part)

    random_words = random.sample(words, length_20percent)

    result = list()
    for word in words:
        if word in random_words:
            continue
        result.append(word)

    return " ".join(result)


def synonym_replacement(text: str, part: float = 0.5, method='word2vec') \
        -> str:  # аугментированный текст

    words = text.split()

    synonyms = list()
    new_words = words.copy()

    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)

    length = len(random_word_list)
    length_20percent = math.trunc(length * part)

    for index, random_word in enumerate(random_word_list):
        if method == 'word2vec':
            # try:
            model = KeyedVectors.load_word2vec_format('../data/220/model.bin', binary=True)
            synonyms = get_neighbors_with_word2vec(random_word, model)
            if len(synonyms) > 0:
                synonym = synonyms[0]
                # print(random_word + '  ' + synonym)
        # except:
        #   continue

        elif method == 'rusvectores':
            synonyms = get_neighbors_with_rusvectores(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(list(synonyms))
                #print(random_word + '  ' + synonym)

        if len(synonyms) > 0:
            ind = new_words.index(random_word)
            new_words[ind] = synonym
            # new_words = [synonym if word == random_word else word for word in new_words]

        if index >= length_20percent:
            break

    return ' '.join(new_words)


def random_insertion(text: str, part: float = 0.5) \
        -> str:  # аугментированный текст

    synonyms = list()

    words = text.split()

    random_word_list = list([word for word in words if word not in stop_words])

    length = len(random_word_list)
    length_20percent = math.trunc(length * part)

    random_words = random.sample(random_word_list, length_20percent)

    for random_word in random_words:
        try:
            model = KeyedVectors.load_word2vec_format('data/220/model.bin', binary=True)
            synonyms = get_neighbors_with_word2vec(random_word, model)

        except:
            continue

        if len(synonyms) > 0:
            random_synonym = synonyms[0]

            index = random.randint(0, len(words) - 1)
            words.insert(index, random_synonym)

    return " ".join(words)


def swap_words(text: str, part: float = 0.3) \
        -> str:  # аугментированный текст

    words = text.split()

    length = len(words)
    length_20percent = math.trunc(length * part)
    indexes = random.sample(range(length), length_20percent)

    for index1 in indexes:
        left_border = max([0, index1 - 1, index1 - 2, index1 - 3, index1 - 4])
        right_border = min([index1 + 4, index1 + 3, index1 + 2, index1 + 1, length - 1])

        random_list = [i for i in range(left_border, right_border + 1)]
        random_list.remove(index1)
        index2 = random.choice(random_list)

        # print('swap: ' + words[index1] + ' ' + words[index2])
        words[index1], words[index2] = words[index2], words[index1]
        # print(' '.join(words))

    return ' '.join(words)
