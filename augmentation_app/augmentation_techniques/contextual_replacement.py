from transformers import pipeline
import random
from typing import List, Tuple


# cointegrated/rubert-tiny2 DeepPavlov/rubert-base-cased bert-base-multilingual-cased
def contextual_replacement(
        texts: List,  # тексты для аугментации
        labels: List,  # метки текстов
        top_k: int = 2,  # топ-k замен выбрать для аугментации каждого сэмпла
        part: float = 0.5,  # замена слов с вероятностью part
        model_name: str = 'cointegrated/rubert-tiny2') \
        -> Tuple[List, List]:  # два списка - новые тексты и их метки

    fill_mask = pipeline('fill-mask', model=model_name)

    new_texts, new_labels = [], []
    # for text, label in tqdm(zip(texts, labels), total=len(texts), desc='contextual_replacer'):
    for text, label in zip(texts, labels):
        split_text = text.split()
        length = len(split_text)

        length_20percent = round(length * part)

        indexes = random.sample(range(length), length_20percent)

        for index in indexes:
            token = split_text[index]
            split_text[index] = '[MASK]'
            # join_text = " ".join(split_text)
            result_replace = fill_mask(" ".join(split_text), top_k=top_k)
            for item in result_replace:
                if item['token_str'] != token:
                    split_text[index] = item['token_str']
                    break
                else:
                    split_text[index] = token

        new_texts.append(' '.join(split_text))
        new_labels.append(label)

    return new_texts, new_labels
