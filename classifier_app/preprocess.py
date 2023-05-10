import re
import urllib
import unicodedata
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import pymorphy3
import pandas as pd
from sklearn.model_selection import train_test_split


morph = pymorphy3.MorphAnalyzer()
file = open("/Users/a.v.protasov/Desktop/diploma/data/stop_words_russian.txt", "r")
stop_words = file.read().splitlines()

def getContentText(url):
    """дополнительная функция для получения текстовых данных
        с тега <body>"""

    userAgent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': userAgent, }
    request = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(request)
    html = response.read()
    soup = BeautifulSoup(html)
    for data in soup(["script", "style"]):
        data.extract()
    allText = soup.body.get_text()
    splitText = allText.splitlines()
    lines = [line.strip() for line in splitText]
    cleanLines = [splitLine.strip() for line in lines for splitLine in line.split("  ")]
    text = "\n".join(line for line in cleanLines if line)

    return unicodedata.normalize("NFKD", text)

def prepare_web_page(url):

    text = getContentText(url)

    return normalize_text(text)



def get_split(text):
    l_total = []
    l_parcial = []
    count = len(text.split()) // 450
    if count > 0:
        n = count
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text.split()[:500]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text.split()[w * 450:w * 450 + 500]
            l_total.append(" ".join(l_parcial))
    return l_total


def normalize_text(text):
    # приведение к нижнему регистру
    lower_result = text.lower()
    # удаление url-адресов
    non_links_result = re.sub(r"\S*https?:\S*", "", lower_result)
    # удаление emails
    non_emails_result = re.sub(r"\S*@\S*\s?", "", non_links_result)
    # удаление цифр
    non_numeric_result = ''.join([i for i in non_emails_result if not i.isdigit()])
    # удаление пунктуации и специальных символов
    non_punc_result = ''.join(
        filter(lambda mark: ord(mark) == 774 or mark.isalnum() or mark.isspace(), non_numeric_result)).strip()
    # удаление лишних пробелов
    non_space_result = re.sub(r" +", " ", non_punc_result)
    # удаление лишних абзацев
    paragraph_list = non_space_result.split('\n')
    text = '\n'.join([p for p in paragraph_list if not p.count(' ') < 10])
    # удаление стоп-слов и длинных слов
    text_tokens = word_tokenize(text)
    # without_stop_word_tokens = [word for word in text_tokens if not word in stop_words]
    # without_long_tokens = [word for word in without_stop_word_tokens if len(word) <= 18]
    without_long_tokens = [word for word in text_tokens if len(word) <= 18]
    without_short_tokens = [word for word in without_long_tokens if len(word) >= 2]
    # лемматизация слов
    # lemmatize_result = [morph.parse(word)[0].normal_form for word in without_short_tokens]
    # filtered_text = (" ").join(lemmatize_result)

    return (" ").join(without_short_tokens)




def prepare_data(path: str, thread=None):

    df = pd.read_csv(path, index_col=0)
    df = df.dropna().reset_index(drop=True)
    category_index = {i[1]: i[0] for i in enumerate(df.category.unique())}
    reverse_category_index = {i[0]: i[1] for i in enumerate(df.category.unique())}

    labels = [category_index[i] for i in df.category.values]
    texts = list()
    if thread:
        thread._signal.emit("Предобработка веб-страниц и их текстового содержимого")
    for index, row in df.iterrows():
        if thread:
            thread._signal.emit(str(round(index / df.shape[0] * 100)))
        texts.append(normalize_text(getContentText(row['url'])))

    if thread:
        thread._signal.emit("100")

    extended_texts = list()
    extended_labels = list()

    if thread:
        thread._signal.emit("Разделение текстов на порции данных")

    for index, (text, label) in enumerate(zip(texts, labels)):
        if thread:
            thread._signal.emit(str(round(index / df.shape[0] * 100)))
        split_text = get_split(text)
        extended_texts.append(split_text)
        extended_labels.append([label] * len(split_text))

    if thread:
        thread._signal.emit("100")

    extended_texts = sum(extended_texts, [])
    extended_labels = sum(extended_labels, [])

    texts = extended_texts
    labels = extended_labels

    texts1 = texts
    texts = []
    labels1 = []

    for text, label in zip(texts1, labels):
        if len(text.split()) >= 20:
            texts.append(text)
            labels1.append(label)

    labels = labels1

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(list(texts), labels, random_state=42, train_size=0.9, stratify=labels)

    return train_texts, valid_texts, train_labels, valid_labels, category_index, reverse_category_index