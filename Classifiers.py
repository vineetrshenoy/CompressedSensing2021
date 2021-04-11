import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # The actual neural network
        self.backbone = backbone

        # Things that need to be saved across training session
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.overall_accuracy = None

    def forward(self, x):
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=0.001) # use for ReddyNet

        optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[15, 30], last_epoch=0 - 1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.training_losses.append(loss.cpu().detach().numpy())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.validation_losses.append(loss.cpu().detach().numpy())
        _, predicted_class = torch.max(y_hat, 1)
        accuracy = torch.sum(predicted_class == y).cpu().detach().numpy() / y.size(0)
        self.validation_accuracies.append(accuracy)
        self.log('val_acc', accuracy, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        _, predicted_class = torch.max(y_hat, 1)
        # Convert to numpy
        y = y.cpu().detach().numpy()
        predicted_class = predicted_class.cpu().detach().numpy()
        # Compute accuracy
        accuracy = np.sum(predicted_class == y) / len(y)
        self.overall_accuracy = accuracy
        print('Overall Accuracy: %.3f' % accuracy)
        # Create a confusion matrix
        cm_normalized = confusion_matrix(y, predicted_class, normalize='true')
        self.cm = cm_normalized

    def test_epoch_end(self, test_step_outputs):
        print("YOYO")
        print(type(test_step_outputs))
        print(len(test_step_outputs))