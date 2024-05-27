from torch import nn, tensor, flatten
import torch.nn.functional as F
from torch import optim
from torchmetrics import F1Score, Accuracy
import lightning as L
from schedulefree import AdamWScheduleFree
from torch import flatten

class Modelo(L.LightningModule):
    def __init__(self, weights):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
             nn.Linear(32768, 512),
             nn.ReLU(inplace=True),
             nn.Dropout(p=0.5),
             nn.Linear(512, 7)
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=7)
        self.f1       =  F1Score(task="multiclass", num_classes=7)
        self.loss     = nn.CrossEntropyLoss(weight=tensor(weights))

        for layer in self.features:
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        #return optim.SGD(self.parameters(), lr = 0.0001, momentum = 0.9, weight_decay = 0.0001)
	    #return optim.Adam(self.parameters(), lr=0.0001)
        #return AdamWScheduleFree(self.parameters(), lr=0.001, betas=(0.95, 0.999), warmup_steps=100)
        return AdamWScheduleFree(self.parameters(), lr=0.001, betas=(0.95, 0.999), warmup_steps=100)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True,prog_bar=True, logger=True)

        accu = self.accuracy(logits, y)
        self.log('train_accu', accu, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        f1_score = self.f1(logits, y)
        self.log('train_f1', f1_score, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)

        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)

        accu = self.accuracy(logits, y)
        self.log('val_accu', accu, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        f1_score = self.f1(logits, y)
        self.log('val_f1', f1_score, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
