import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    AutoImageProcessor,
)
import evaluate
import numpy as np


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_hidden=1, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(d_in, d_hidden)
        self.layers = nn.Sequential(
            *(
                [
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                ]
                * n_hidden
            )
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        x = torch.flatten(pixel_values, 1)
        x = self.input_linear(x)
        x = self.layers(x)
        x = self.output_layer(x)
        preds = torch.softmax(x, dim=-1)
        # Compute loss.
        if labels is not None:
            loss = self.loss_fn(preds, labels)
            return (loss, preds)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        pass


accuracy = evaluate.load("accuracy")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(model_cls, dataset, train_args, **model_params):
    model = model_cls(**model_params)
    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
