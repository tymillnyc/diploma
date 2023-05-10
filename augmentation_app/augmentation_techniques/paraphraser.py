from typing import List, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer


def paraphraser(
        texts: List,  # тексты для перефразировки
        labels: List,  # метки текстов
        grams: int = 5,  # 5-8
        beams_number: int = 1,  # количество лучей для генерации новой последовательности
        sequences_number: int = 3,  # количество новых текстов, которое необходимо сгенерировать для каждого сэмпла
        top_k: int = 150,  # 50-150
        top_p: float = 0.8,  # сохраняются только наиболее вероятные токены, sum(p) > top_p. 0.6-0.9
        model_name: str = 'cointegrated/rut5-base-paraphraser') \
        -> Tuple[List, List]:  # два списка - новые тексты и их метки

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # model.cuda()
    model.eval()

    new_texts, new_labels = [], []
    # for text, label in tqdm(zip(texts, labels), total=len(texts), desc='paraphraser'):
    for text, label in zip(texts, labels):
        x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        max_size = int(x.input_ids.shape[1] * 1.5 + 10)

        out = model.generate(**x,
                             encoder_no_repeat_ngram_size=grams,
                             num_beams=beams_number,
                             max_length=max_size,
                             num_return_sequences=sequences_number,
                             do_sample=True,
                             top_k=top_k,
                             top_p=top_p)
        # temperature=1.5)

        result_texts = tokenizer.batch_decode(out, skip_special_tokens=True)

        for t in set(result_texts):
            if t != text:
                new_texts.append(t)
                new_labels.append(label)

    return new_texts, new_labels
