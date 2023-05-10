import os
import pickle
import random as rand
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import CustomDataset
import sentencepiece, sacremoses, subword_nmt, importlib_metadata

class BertClassifier:

    def __init__(self, model_path=None, tokenizer_path=None, category_index=None, reverse_category_index=None, epochs=4,
                 model_save_path='models/'):

        if not category_index:

            self.config = AutoConfig.from_pretrained(tokenizer_path)

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, pad_to_max_length=True)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=self.config)

        else:

            config = AutoConfig.from_pretrained(model_path,
                                                num_labels=len(category_index),
                                                id2label=reverse_category_index,
                                                label2id=category_index)


            with open(model_save_path + 'saved_dictionary.pkl', 'wb') as f:
                pickle.dump(reverse_category_index, f)

            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', tokenizer_path)

            self.model_save_path = model_save_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_len = 500
        self.epochs = epochs

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        batch_size = 16

        self.train_dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        self.validation_dataloader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=True)

        self.named_parameters = list(self.model.named_parameters())

        self.no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        self.grouped_parameters = [
            {'params': [p for name, p in self.named_parameters if not any(nd in name for nd in self.no_decay)],
             'weightDecay': 0.01},  # param weight
            {'params': [p for name, p in self.named_parameters if any(nd in name for nd in self.no_decay)],
             'weightDecay': 0.0}  # param bias, ...
        ]

        self.optimizer = torch.optim.AdamW(self.grouped_parameters, lr=2e-5)

    def fit(self, app=None):
        self.model = self.model.train()

        total_loss = 0

        train_preds, train_labels = list(), list()

        for step, batch in enumerate(self.train_dataloader):

            # переводим данные на видеокарту
            b_input_ids = batch["input_ids"].to(self.device)
            b_input_mask = batch["attention_mask"].to(self.device)
            b_labels = batch["targets"].to(self.device)

            # обнуляем градиенты
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            loss = outputs[0]

            logits = outputs[1]

            total_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            batch_preds = np.argmax(logits, axis=1)
            train_preds.extend(batch_preds)
            train_labels.extend(label_ids)

            loss.backward()

            self.optimizer.step()

            if (step) % 20 == 0 and not step == 0:
                app.threePageLogs.insertPlainText('  Step {:>3,} of {}. Loss: {:.4f}'.format(step, len(self.train_dataloader), loss.item()))
                app.threePageLogs.insertPlainText('\n')

        avg_train_loss = total_loss / len(self.train_dataloader)
        train_f1_micro = f1_score(train_labels, train_preds, average="micro")

        return train_f1_micro, avg_train_loss

    def eval(self):
        self.model = self.model.eval()

        total_eval_loss = 0

        valid_preds, valid_labels = list(), list()

        for batch in self.validation_dataloader:
            b_input_ids = batch["input_ids"].to(self.device)
            b_input_mask = batch["attention_mask"].to(self.device)
            b_labels = batch["targets"].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            logits = outputs[1]

            loss = outputs[0]

            total_eval_loss += loss.item()

            # перемещаем логиты и метки на CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            batch_preds = np.argmax(logits, axis=1)
            valid_preds.extend(batch_preds)
            valid_labels.extend(label_ids)

        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        valid_f1_micro = f1_score(valid_labels, valid_preds, average="micro")

        return avg_val_loss, valid_f1_micro

    def train(self, thread=None, app=None):

        best_f1_micro = 0

        train_loss_values = list()
        train_f1_micro_values = list()
        valid_loss_values = list()
        valid_f1_micro_values = list()

        for epoch in range(self.epochs):

            if thread:
                thread._signal.emit("Обучение (эпоха  " + str(epoch+1) + '/' + str(self.epochs) + ')')
                thread._signal.emit(str(round(epoch / self.epochs * 100)))

            app.threePageLogs.insertPlainText('Training...\n')

            train_f1_micro, avg_train_loss = self.fit(app)

            train_loss_values.append(avg_train_loss)
            train_f1_micro_values.append(train_f1_micro)

            app.threePageLogs.insertPlainText('\n')
            app.threePageLogs.insertPlainText('  Training f1-micro: {0:.2%}'.format(train_f1_micro))
            app.threePageLogs.insertPlainText('\n')
            app.threePageLogs.insertPlainText('  Training loss: {0:.4f}'.format(avg_train_loss))
            app.threePageLogs.insertPlainText('\n')
            app.threePageLogs.insertPlainText('Running Validation...\n')

            avg_val_loss, valid_f1_micro = self.eval()

            valid_loss_values.append(avg_val_loss)
            valid_f1_micro_values.append(valid_f1_micro)

            app.threePageLogs.insertPlainText('  F1-micro: {0:.2%}'.format(valid_f1_micro))
            app.threePageLogs.insertPlainText('\n')
            app.threePageLogs.insertPlainText('   Validation loss: {0:.4f}'.format(avg_val_loss))

            if valid_f1_micro > best_f1_micro:
                best_f1_micro = valid_f1_micro
                self.save_model(self.model, self.tokenizer, epoch, app)

        if thread:
            thread._signal.emit("100")


    def save_model(self, model, tokenizer, epoch=None, app=None):
        app.threePageLogs.insertPlainText(f'\nSaving on epoch {epoch + 1}')
        app.threePageLogs.insertPlainText('\n')
        self.directory = self.model_save_path + 'model_bert' + str(rand.randint(1, 50000)) + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        tokenizer.save_pretrained(self.directory)
        model.save_pretrained(self.directory)

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction