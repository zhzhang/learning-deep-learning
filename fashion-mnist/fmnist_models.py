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
    def __init__(self, d_in, d_hidden, d_out, n_hidden=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(d_in, d_hidden)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(d_hidden, d_hidden))
        self.layers = nn.Sequential(
            *layers,
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
    def __init__(self, d_out=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3200, d_out)

    def forward(self, pixel_values, labels=None):
        x = self.conv1(pixel_values)
        x = torch.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.max_pool2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        preds = torch.softmax(x, dim=-1)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(preds, labels)
            return (loss, preds)
        return (preds,)


accuracy = evaluate.load("accuracy")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(model_cls, dataset, train_args, **model_params):
    model = model_cls(**model_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params}")
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
    return trainer
