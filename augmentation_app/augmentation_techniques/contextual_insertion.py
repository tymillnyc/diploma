from transformers import pipeline
import random
import math
from typing import List, Tuple


# cointegrated/rubert-tiny2 DeepPavlov/rubert-base-cased bert-base-multilingual-cased
def contextual_insertion(
        texts: List,  # тексты для аугментации
        labels: List,  # метки текстов
        top_k: int = 2,  # топ-k замен выбрать для аугментации каждого сэмпла
        part: float = 0.5,  # вставка слов с вероятностью part
        model_name: str = 'cointegrated/rubert-tiny2') \
        -> Tuple[List, List]:  # два списка - новые тексты и их метки

    fill_mask = pipeline('fill-mask', model=model_name)

    new_texts, new_labels = [], []
    # for text, label in tqdm(zip(texts, labels), total=len(texts), desc='contextual_insertion'):
    for text, label in zip(texts, labels):
        split_text = text.split()
        with_mask_text = list()

        length = len(split_text)
        length_20percent = math.trunc(length * part)

        indexes = sorted(random.sample(range(length), length_20percent))

        for i in range(length):
            if i in indexes:
                with_mask_text.append('[MASK]')
            with_mask_text.append(split_text[i])

        indexes = [i for i in range(len(with_mask_text)) if with_mask_text[i] == '[MASK]']

        for index in indexes:
            non_masks = list()
            for i in range(index + 1, len(with_mask_text)):
                if with_mask_text[i] == '[MASK]':
                    continue
                non_masks.append(with_mask_text[i])
            with_first_mask = with_mask_text[0:index + 1] + non_masks
            result_replace = fill_mask(" ".join(with_first_mask), top_k=top_k)
            for item in result_replace:
                with_mask_text[index] = item['token_str']
                break

        new_texts.append(' '.join(with_mask_text))
        new_labels.append(label)

    return new_texts, new_labels
