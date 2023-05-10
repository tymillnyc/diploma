from transformers import pipeline, BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple
from deep_translator import GoogleTranslator


def back_translation(
        texts: List,  # тексты для аугментации
        labels: List,  # метки текстов
        src: str = 'ru',  # язык, с которого переводить
        tgt: str = 'en',  # язык, на который переводить
        batch_size: int = 20) \
        -> Tuple[List, List]:  # два списка - новые тексты и их метки

    new_texts, new_labels = [], []
    text = texts[0]

    try:
        z = 1/0

        tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}")

        # model.cuda()

        input_ids = tokenizer.encode(text, return_tensors="pt")  # .to(model.device)
        outputs = model.generate(input_ids, max_length=500)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}")
        model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}")

        # model.cuda()

        input_ids = tokenizer.encode(decoded, return_tensors="pt")  # .to(model.device)
        outputs = model.generate(input_ids, max_length=500)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded != text:
            new_texts.append(decoded)
            new_labels.append(labels[0])

        # translator = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{src}-{tgt}')
        # translator_back = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{tgt}-{src}')

        # for step in trange(0, len(texts), batch_size, desc='back_translation'):
        # for step in range(0, len(texts), batch_size):

        # texts[step:step+batch_size] -> texts[step], labels[step:step+batch_size] -> labels[step]

        # transleted = [t['translation_text'] for t in translator(texts[step:step+batch_size], truncation='only_first')]
        # back = [t['translation_text'] for t in translator_back(transleted, truncation='only_first')]

        # for t, b, l in zip(texts[step:step+batch_size], back, labels[step:step+batch_size]):
        # if t != b:
        #   new_texts.append(b)
        #   new_labels.append(l)
    except:
        try:
            translated = GoogleTranslator(source='ru', target='en').translate(text)
            back = GoogleTranslator(source='en', target='ru').translate(translated)

            if back != text:
                new_texts.append(back)
                new_labels.append(labels[0])
        except:
            print("Error not translated.")
            return [''], [-1]

    return new_texts, new_labels
