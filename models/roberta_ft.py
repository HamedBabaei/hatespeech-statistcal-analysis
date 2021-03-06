"""
    RoBERTa Classifier: fine tuning and predictions
"""
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
import numpy as np
from pprint import pprint
import os
from datahandlers import DataWriter
from .evaluation import evaluate


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


class RoBERTaHSDetector:
    """
        RoBERTa Classifier For Hate Speech Identification
    """
    def __init__(self, model_path, tokenizer_path, max_length, device):
        self.device = device
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(model_path) #, from_tf=False
        self.roberta_model.to(self.device)
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_length=max_length)

    def fine_tune(self, config, train, val, test, metrics):
        def tokenization(batched_text):
            return self.roberta_tokenizer(batched_text['text'], padding=True, truncation=True)

        dataset_train = datasets.Dataset.from_dict(train)
        dataset_val = datasets.Dataset.from_dict(val)
        dataset_test = datasets.Dataset.from_dict(test)

        dataset_train = dataset_train.map(tokenization, batched=True, batch_size=len(train))
        dataset_val = dataset_val.map(tokenization, batched=True, batch_size=len(val))
        dataset_test = dataset_test.map(tokenization, batched=True, batch_size=len(test))

        dataset_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
                                config.roberta_fine_tune,
                                save_total_limit=config.save_total_limit,
                                evaluation_strategy="epoch",
                                per_device_train_batch_size=config.batch_size,
                                per_device_eval_batch_size=config.batch_size,
                                num_train_epochs=config.epoch,
                                weight_decay=config.weight_decay,
                                logging_steps=300,
                                save_steps=-1
                                )
        self.trainer = Trainer(
                    model=self.roberta_model,
                    args=training_args,
                    compute_metrics=metrics,
                    train_dataset=dataset_train,
                    eval_dataset=dataset_val
        )

        self.trainer.train()
        self.trainer.save_model(config.roberta_fine_tune)
        pprint(self.trainer.evaluate())

        val_predicts = self.trainer.predict(dataset_val)
        val_labels = val_predicts.label_ids
        val_preds = val_predicts.predictions.argmax(-1)
        val_f1, val_acc, val_clf_report = evaluate(gold=val_labels,
                                                   predicts=val_preds,
                                                   average='macro')

        test_predicts = self.trainer.predict(dataset_test)
        test_labels = test_predicts.label_ids
        test_preds = test_predicts.predictions.argmax(-1)
        test_f1, test_acc, test_clf_report = evaluate(gold=test_labels,
                                                      predicts=test_preds,
                                                      average='macro')


        report = {
            "Test-F1 Macro": test_f1,
            "Test-accuracy": test_acc,
            "Test-classification-report": test_clf_report,
            "Test-gt": [int(y) for y in list(test_labels)],
            "Test-predict": [int(pred) for pred in list(test_preds)],
            "Val-F1 Macro": val_f1,
            "Val-accuracy": val_acc,
            "Val-classification-report": val_clf_report,
            "Val-gt": [int(y) for y in list(val_labels)],
            "Val-predict": [int(pred) for pred in list(val_preds)]
        }

        path_to_report = os.path.join(config.logs_dir, config.dataset + "-evaluation-" + config.model_name + ".json")
        print(f"Save results with gt and predicts into :{path_to_report}")
        DataWriter.write_json(report, path_to_report)

        # Keep track of train and evaluate loss.
        history = {
            "train_loss": [], "eval_loss": [],
            "train_f1":[], "eval_f1": [],
            "start_step": training_args.logging_steps,
            "step_size": training_args.logging_steps
        }
        for log_history in self.trainer.state.log_history:
            if 'loss' in log_history.keys():
                history['train_loss'].append(log_history['loss'])
            elif 'eval_loss' in log_history.keys():
                history['eval_loss'].append(log_history['eval_loss'])
            elif 'f1' in log_history.keys():
                history['f1'].append(log_history['f1'])
            elif 'eval_f1' in log_history.keys():
                history['eval_f1'].append(log_history['eval_f1'])

        path_to_log_report = os.path.join(config.logs_dir, config.dataset + "-history-" + config.model_name + ".json")
        DataWriter.write_json(history, path_to_log_report)


    def _predict(self, X:str, proba:bool = False):
        inputs = self.roberta_tokenizer(X, return_tensors="pt").to(self.device)
        outputs = self.roberta_model(**inputs)
        logits = outputs.logits
        logits = logits.to("cpu").detach().numpy()
        if proba:
            return logits
        else:
            return np.argmax(logits)

    def predict(self, X:list, proba:bool = False):
        tokenized_texts = self.roberta_tokenizer(X, padding=True, truncation=True)
        trainer = Trainer(model=self.roberta_model)
        test_dataset = SimpleDataset(tokenized_texts)
        predictions = trainer.predict(test_dataset)
        return predictions.predictions.argmax(-1)
