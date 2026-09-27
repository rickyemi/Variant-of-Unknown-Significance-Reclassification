"""
FT-Transformer (Feature Tokenizer + Transformer) for tabular data, wrapped as a
scikit-learn classifier so it can sit inside a Pipeline, be cross-validated,
permutation-tested and saved with joblib exactly like XGBoost and the SVM.

Architecture (Gorishniy et al., 2021, "Revisiting Deep Learning Models for
Tabular Data")
-------------------------------------------------------------------------
* Feature tokenizer: every input column j becomes a d-dimensional token
  x_j * W_j + b_j, so each feature has its own learned embedding.
* A learnable [CLS] token is prepended.
* L pre-norm Transformer encoder blocks (multi-head self-attention + MLP)
  let features attend to each other, capturing interactions.
* The final [CLS] representation goes through LayerNorm -> ReLU -> Linear to
  one logit; sigmoid gives P(Pathogenic).

Training: AdamW, binary cross-entropy, mini-batches, and early stopping on a
stratified 15% validation slice of the training data (best weights restored).
"""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn


class _FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=d_token ** -0.5)
        nn.init.normal_(self.bias, std=d_token ** -0.5)

    def forward(self, x):                      # x: (batch, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias   # (batch, n_features, d)


class _FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=2,
                 ff_mult=2, dropout=0.2):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0]).squeeze(-1)   # one logit per row


class FTTransformerClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn compatible FT-Transformer binary classifier.

    Expects a dense, already preprocessed (imputed, scaled, one-hot) matrix,
    which is what the project's preprocessing pipeline produces.
    """

    def __init__(self, d_token=32, n_heads=4, n_layers=2, ff_mult=2, dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200,
                 patience=20, val_fraction=0.15, random_state=42, verbose=False):
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state)
        Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xva_t = torch.from_numpy(Xva)

        self.model_ = _FTTransformer(X.shape[1], self.d_token, self.n_heads,
                                     self.n_layers, self.ff_mult, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.random_state)

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history_ = {"train_loss": [], "val_loss": [], "val_auc": []}
        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(Xtr_t), generator=gen)
            running = 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                running += loss.item() * len(idx)
            # validation
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(Xva_t)
                v_loss = loss_fn(logits, torch.from_numpy(yva)).item()
                v_auc = roc_auc_score(yva, torch.sigmoid(logits).numpy())
            self.history_["train_loss"].append(running / len(Xtr_t))
            self.history_["val_loss"].append(v_loss)
            self.history_["val_auc"].append(v_auc)
            if self.verbose and epoch % 10 == 0:
                print(f"epoch {epoch:3d} train {running / len(Xtr_t):.4f} "
                      f"val {v_loss:.4f} auc {v_auc:.4f}")
            if v_loss < best_loss - 1e-4:
                best_loss, bad_epochs = v_loss, 0
                best_state = copy.deepcopy(self.model_.state_dict())
                self.best_epoch_ = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                out.append(torch.sigmoid(self.model_(X[i:i + 1024])).numpy())
        p1 = np.concatenate(out) if out else np.empty(0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]
