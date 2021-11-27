"""
    RoBERTa Classifier: fine tuning and predictions
"""
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
import numpy as np
from pprint import pprint
from tqdm import tqdm
import torch
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

    def fine_tune(self, config, train, val, metrics):
        def tokenization(batched_text):
            return self.roberta_tokenizer(batched_text['text'], padding=True, truncation=True)

        dataset_train = datasets.Dataset.from_dict(train)
        dataset_val = datasets.Dataset.from_dict(val)

        dataset_train = dataset_train.map(tokenization, batched=True, batch_size=len(train))
        dataset_val = dataset_val.map(tokenization, batched=True, batch_size=len(val))

        dataset_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
                                config.roberta_fine_tune,
                                save_total_limit=config.save_total_limit,
                                evaluation_strategy="epoch",
                                per_device_train_batch_size=config.batch_size,
                                per_device_eval_batch_size=config.batch_size,
                                num_train_epochs=config.epoch,
                                weight_decay=config.weight_decay)
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
        predicts = self.trainer.predict(dataset_val)

        labels = predicts.label_ids
        preds = predicts.predictions.argmax(-1)

        f1, acc, clf_report = evaluate(gold=labels,
                                       predicts=preds,
                                       average='macro')
        report = {"F1 Macro": f1, "accuracy": acc, "classification-report": clf_report}

        pprint(report)

        path_to_report = os.path.join(config.logs_dir, config.dataset + "-evaluation-" + config.model_name + ".json")
        print(f"Save results with gt and predicts into :{path_to_report}")
        report["gt"], report['predict'] = [int(label) for label in list(labels)], [int(pred) for pred in list(preds)]
        DataWriter.write_json(report, path_to_report)


    def __predict(self, X:str, proba:bool = False):
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

    def predict2(self, X:list):
        preds = []
        self.roberta_model.eval()
        for x in tqdm(X):
            tokenized_texts = self.roberta_tokenizer([x], padding=True, truncation=True)
            pt_inputs = {k: torch.tensor(v).to(self.device) for k, v in tokenized_texts.items()}
            with torch.no_grad():
                output = self.roberta_model(**pt_inputs)
            predictions = output.logits.cpu().numpy()
            preds.append(predictions.argmax(-1)[0])
        return preds
