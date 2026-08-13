import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm


class FullyConnectedNN(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.threshold = threshold

        self.model = nn.Sequential(
            nn.Linear(17, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        probs = self.predict_proba(x)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            output = self(x)
            return torch.sigmoid(output).cpu().numpy()

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs=100,
        learning_rate=0.001,
        log_to_mlflow=True,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

        if log_to_mlflow and mlflow.active_run():
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "optimizer": "AdamW",
                    "scheduler": "OneCycleLR",
                    "threshold": self.threshold,
                    "batch_size": train_loader.batch_size,
                }
            )

        f1_scores = []
        roc_auc_scores = []

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            target = []
            pred = []
            for X_batch, y_batch in tqdm(
                train_loader, leave=False, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                train_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            train_loss /= len(train_loader)

            epoch_metrics = {"train/loss": train_loss}

            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch_val, y_batch_val in tqdm(
                        val_loader,
                        leave=False,
                        desc=f"Validation Epoch {epoch + 1}/{epochs}",
                    ):
                        y_pred = self(X_batch_val)
                        loss = self.loss_fn(y_pred, y_batch_val)
                        val_loss += loss.item()

                        target.append(y_batch_val.numpy())
                        pred.append(y_pred.numpy())

                target_np = np.vstack(target)
                pred_np = np.vstack(pred)

                pred_np_probs = 1 / (1 + np.exp(-pred_np))
                pred_np_binary = (pred_np_probs >= self.threshold).astype(int)

                f1 = f1_score(
                    target_np,
                    pred_np_binary,
                    average="weighted",
                    zero_division=0,
                )

                roc_auc = roc_auc_score(target_np, pred_np_probs)

                f1_scores.append(f1)
                roc_auc_scores.append(roc_auc)
                val_loss /= len(val_loader)

                epoch_metrics.update(
                    {
                        "val/loss": val_loss,
                        "val/f1_weighted": f1,
                        "val/roc_auc": roc_auc,
                    }
                )

                print(
                    f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Training Loss: {train_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")

            if log_to_mlflow and mlflow.active_run():
                mlflow.log_metrics(epoch_metrics, step=epoch + 1)

        targets_np = np.vstack(target)
        preds_np = np.vstack(pred)

        preds_probs = 1 / (1 + np.exp(-preds_np))
        preds_binary = (preds_probs >= self.threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(targets_np, preds_binary),
            "f1": f1_score(
                targets_np, preds_binary, average="weighted", zero_division=0
            ),
            "precision": precision_score(targets_np, preds_binary),
            "recall": recall_score(targets_np, preds_binary),
            "roc_auc": roc_auc_score(targets_np, preds_probs),
        }

        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metrics(metrics)

        return metrics, f1_scores, roc_auc_scores


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):

        x_values = X.values if hasattr(X, "values") else X
        y_values = y.values if hasattr(y, "values") else y

        self.X = torch.tensor(x_values, dtype=torch.float32)
        self.y = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)

        if len(self.y.shape) == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
