import re
from typing import List, Tuple
import nltk
from pymystem3 import Mystem
import requests

ru_rnc = {'A': 'ADJ', 'ADV': 'ADV', 'ADVPRO': 'ADV',
          'ANUM': 'ADJ', 'APRO': 'DET', 'COM': 'ADJ',
          'CONJ': 'SCONJ', 'INTJ': 'INTJ', 'NONLEX': 'X',
          'NUM': 'NUM', 'PART': 'PART', 'PR': 'ADP',
          'S': 'NOUN', 'SPRO': 'PRON', 'UNKN': 'X', 'V': 'VERB'}

ru_un = {'!': '.', 'A': 'ADJ', 'C': 'CONJ', 'AD': 'ADV', 'NNS': 'NOUN', 'NNP': 'NOUN',
         'NN': 'NOUN', 'VG': 'VERB', 'COMP': 'CONJ',
         'NC': 'NUM', 'VP': 'VERB', 'P': 'ADP',
         'IJ': 'X', 'V': 'VERB', 'Z': 'X', 'VI': 'VERB', 'YES_NO_SENT': 'X', 'PTCL': 'PRT',
         'VBP': 'VBP', 'VBN': 'VERB', 'VBG': 'VERB', 'VBD': 'VERB', 'VB': 'VERB'}

file = open("../data/stop_words_russian.txt", "r")
stop_words = file.read().splitlines()

def check_english(word):
    return bool(re.search('[a-zA-Z]', word))


def get_tag(word) \
        -> Tuple[str, str]:  # лексема и тэг слова word

    if check_english(word):
        lex = word
        try:
            pos = ru_un[nltk.pos_tag([word])[0][1]]
        except:
            return word, 'NOUN'
    else:
        m = Mystem()
        processed = m.analyze(word)[0]

        lex = processed["analysis"][0]["lex"].lower().strip()
        pos = processed["analysis"][0]["gr"].split(',')[0]
        pos = pos.split('=')[0].strip()

    return lex, pos


def get_neighbors_with_rusvectores(
        word: str,
        model: str = 'ruwikiruscorpora_upos_cbow_300_10_2021',
        format_: str = 'csv') \
        -> List:  # список самых 'близких' слов к word

    lex, pos = get_tag(word)
    # tag = word_tag.split('_', 1)[1]
    if not check_english(word):
        pos = ru_rnc[pos]

    word = lex + '_' + pos

    neighbors = list()

    url = '/'.join(['https://rusvectores.org', model, word, 'api', format_]) + '/'
    r = requests.get(url=url, stream=True)

    for line in r.text.split('\n'):
        try:
            token, sim = re.split('\s+', line)
            tag = token.split('_', 1)[1]
            word = token.split('_', 1)[0]

            if word != lex and tag == pos:
                neighbors.append(word)
        except:
            continue

    return neighbors


def get_neighbors_with_word2vec(
        word: str,
        model) \
        -> List:  # список самых 'близких' слов к word

    lex, pos = get_tag(word)
    if not check_english(word):
        pos = ru_rnc[pos]

    word = lex + '_' + pos

    try:
        # model = KeyedVectors.load_word2vec_format('220/model.bin', binary=True)
        neighbors_sim = model.most_similar(word)
    except:
        return []

    tokens = [word for (word, sim) in neighbors_sim]

    neighbors = list()
    for token in tokens:
        tag = token.split('_', 1)[1]
        word = token.split('_', 1)[0]

        if word != lex and tag == pos:
            neighbors.append(word)

    return neighbors
