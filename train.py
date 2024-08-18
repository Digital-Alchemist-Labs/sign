import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from model_dim import skeleTransLayer as skelTrans2
from dataset import OurDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import yaml
import matplotlib.pyplot as plt


def custom_collate(batch):
    input_list = [item["input"] for item in batch]
    labels = [item["label"] for item in batch]
    len_list = [item["frame_len"] for item in batch]
    input_list = pad_sequence(input_list, batch_first=True, padding_value=0.0)
    return {"input": input_list.contiguous(), "label": labels, "len_list": len_list}


def gen_src_mask(total_len, len_list):
    batch_len = len(len_list)  # 리스트의 길이를 구합니다.
    zero = torch.zeros(batch_len, total_len + 1)
    for tens, t in zip(zero, len_list):
        mask = torch.ones(total_len - t)
        tens[t + 1:] = mask
    ret = zero.bool()
    return torch.transpose(ret, 0, 1)


class TrainerModule(pl.LightningModule):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.model1 = skelTrans2(self.num_classes, 21, 1, 60, 1, mask=False)
        self.model2 = skelTrans2(self.num_classes, 21, 1, 60, 1, mask=False)
        self.model3 = skelTrans2(self.num_classes, 42, 1, 60, 1, mask=False)
        
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.train_loss_history = []
        self.val_acc_history = []

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["label"]
        l = batch["len_list"]
        _, total_len, _, _ = x.shape
        mask = gen_src_mask(total_len, l).to("cuda")
        labels = torch.as_tensor(y, device='cuda')
        z = self.model1(x[:, :, :21, :], mask)
        z2 = self.model2(x[:, :, 21:, :], mask)
        z3 = self.model3(x, mask)
        loss = F.cross_entropy(z, labels)
        loss2 = F.cross_entropy(z2, labels)
        loss3 = F.cross_entropy(z3, labels)
        loss_sum = (loss + loss2 + loss3) / 3
        acc = self.accuracy(z, labels)
        
        self.train_loss_history.append(loss_sum.item())
        self.log("train_loss_step", loss_sum)
        self.log("train_acc_step", acc)
        return loss_sum

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["label"]
        l = batch["len_list"]

        with torch.no_grad():
            labels = torch.as_tensor(y, device='cuda')
            z = self.model1(x[:, :, :21, :], None)
            z2 = self.model2(x[:, :, 21:, :], None)
            z3 = self.model3(x, None)
            z = (z + z2 + z3) / 3

            acc = self.accuracy(z, labels)
            self.val_acc_history.append(acc.item())
            self.log("val_acc_step", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def on_train_end(self):
        self.plot_metrics()

    def plot_metrics(self):
        epochs = range(len(self.train_loss_history))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_acc_history, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.keyp_path = "./our_data"
        self.batch_size = batch_size
        self.num_workers = min(15, torch.multiprocessing.cpu_count())

    def setup(self, stage=None):
        self.nia_train = OurDataset(keyp_path=self.keyp_path)
        self.nia_val = OurDataset(keyp_path=self.keyp_path)

    def train_dataloader(self):
        return DataLoader(
            self.nia_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.nia_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["num_classes"]
    epochs = cfg["epochs"]
    ckpt_name = cfg["ckpt_name"]

    trainer = pl.Trainer(accelerator="gpu", max_epochs=epochs, devices=1)
    model = TrainerModule(num_classes=num_classes)
    data = DataModule()
    trainer.fit(model, data)
    trainer.save_checkpoint(ckpt_name)
