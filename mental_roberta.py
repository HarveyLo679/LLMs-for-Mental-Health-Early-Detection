import os
import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr


MODEL_NAME = "mental/mental-roberta-base"
MAX_LEN = 256
BATCH_ENC = 32        
SEED = 41     
USE_CLS = False        

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load & split
def load_posts_csv(path:str,
                   cols=('user_id','timestamp','week','clean_text','valence_w','arousal_w','radius_w'),
                   min_tokens:int=5) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[list(cols)].dropna().reset_index(drop=True)
    df['clean_text'] = df['clean_text'].astype(str).str.strip()
    # simple token count filter
    df['__tok'] = df['clean_text'].str.split().apply(len)
    df = df[df['__tok'] >= min_tokens].reset_index(drop=True)
    df = df.drop(columns='__tok')
    return df

def split_by_user(df: pd.DataFrame,
                  test_frac=0.15, val_frac=0.10, random_state=42):
    users = df['user_id'].dropna().unique().tolist()
    rng = np.random.default_rng(random_state)
    rng.shuffle(users)

    n_users = len(users)
    n_test = round(test_frac * n_users)
    n_trainval = n_users - n_test
    n_val = round(val_frac * n_trainval)
    n_train = n_trainval - n_val

    test_users = set(users[:n_test])
    val_users  = set(users[n_test:n_test+n_val])
    train_users= set(users[n_test+n_val:])

    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df   = df[df['user_id'].isin(val_users)].reset_index(drop=True)
    test_df  = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    return train_df, val_df, test_df

# Tokenize and encode

class _TextDS(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]

def _collate_pad(batch, tok):
    return tok(batch,
               truncation=True,
               max_length=MAX_LEN,
               padding=True,
               return_attention_mask=True,
               return_tensors="pt")

@torch.no_grad()
def encode_texts(texts, batch_size=BATCH_ENC, use_cls=False):
    """
    Returns mean-pooled (default) or CLS embeddings, shape (N, H=768).
    Encoder is frozen & in eval() mode.
    """
    device = DEVICE
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    enc = AutoModel.from_pretrained(MODEL_NAME).to(device)
    enc.eval()

    ds = _TextDS(list(map(str, texts)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: _collate_pad(b, tok))

    chunks = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = enc(**batch)
        last_hidden = out.last_hidden_state  # (B, T, H)

        if use_cls:
            emb = last_hidden[:, 0, :]       # CLS
        else:
            # mean-pool over valid tokens
            mask = batch["attention_mask"].unsqueeze(-1)     # (B, T, 1)
            masked = last_hidden * mask                      # (B, T, H)
            counts = mask.sum(dim=1)                         # (B, 1, 1)->(B,1,1)
            emb = masked.sum(dim=1) / counts                 # (B, H)

        chunks.append(emb.cpu())

    return torch.cat(chunks, dim=0).numpy()


# Baseline: RidgeCV

def ridge_baseline(X_tr, Y_tr, X_va, Y_va, X_te, Y_te):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_va)
    Xte = scaler.transform(X_te)

    alphas = np.logspace(-2, 3, 1000)
    rv = RidgeCV(alphas=alphas)
    ra = RidgeCV(alphas=alphas)

    rv.fit(Xtr, Y_tr[:,0])
    ra.fit(Xtr, Y_tr[:,1])

    pred_val = np.column_stack([rv.predict(Xva), ra.predict(Xva)])
    pred_te  = np.column_stack([rv.predict(Xte), ra.predict(Xte)])

    return metrics_dict(Y_va, pred_val), metrics_dict(Y_te, pred_te)


# MLP head

class MLP(nn.Module):
    def __init__(self, d_in, h=512, p=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, 2)
        )
    def forward(self, x): return self.net(x)

@dataclass
class MLPConfig:
    h: int = 512
    p: float = 0.05
    lr: float = 1e-3
    wd: float = 0.0
    loss: str = "huber"     # "huber" or "mse"
    epochs: int = 12
    batch: int = 256
    patience: int = 3
    seed: int = SEED

def train_mlp_return_preds(X_tr, Y_tr, X_va, Y_va, X_te, cfg: MLPConfig):
    set_seed(cfg.seed)
    device = DEVICE

    X_train = torch.from_numpy(X_tr).float().to(device)
    Y_train = torch.from_numpy(Y_tr).float().to(device)
    X_val = torch.from_numpy(X_va).float().to(device)
    Y_val = torch.from_numpy(Y_va).float().to(device)
    X_test = torch.from_numpy(X_te).float().to(device)

    model = MLP(X_train.shape[1], h=cfg.h, p=cfg.p).to(device)
    opt   = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lossf = nn.SmoothL1Loss() if cfg.loss == "huber" else nn.MSELoss()

    best = {"mae_sum": float("inf"), "state": None}
    no_imp = 0

    for ep in range(1, cfg.epochs+1):
        model.train()
        idx = torch.randperm(X_train.size(0), device=device)
        for i in range(0, X_train.size(0), cfg.batch):
            b = idx[i:i+cfg.batch]
            opt.zero_grad(set_to_none=True)
            pred = model(X_train[b])
            loss = lossf(pred, Y_train[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            p_val = model(X_val).cpu().numpy()
        m = metrics_dict(Y_val, p_val)
        if m["mae_sum"] < best["mae_sum"] - 1e-5:
            best["mae_sum"] = m["mae_sum"]
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp > cfg.patience:
                break

    # reload best and predict val/test
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        Y_val_pred  = model(X_val).cpu().numpy()
        Y_test_pred = model(X_test).cpu().numpy()

    return model, Y_val_pred, Y_test_pred


# Metrics & diagnostics
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def metrics_dict(y_true, y_pred):

    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)

    v_t, a_t = y_true[:,0], y_true[:,1]
    v_p, a_p = y_pred[:,0], y_pred[:,1]

    out = {
        "mae_valence": mean_absolute_error(v_t, v_p),
        "mae_arousal": mean_absolute_error(a_t, a_p),
        "r_valence": pearsonr(v_t, v_p)[0] if len(v_t)>2 else np.nan,
        "r_arousal": pearsonr(a_t, a_p)[0] if len(a_t)>2 else np.nan,
        "rho_valence": spearmanr(v_t, v_p).correlation if len(v_t)>2 else np.nan,
        "rho_arousal": spearmanr(a_t, a_p).correlation if len(a_t)>2 else np.nan,
    }
    out["mae_sum"] = out["mae_valence"] + out["mae_arousal"]
    return out

def calibration(y_true, y_pred):
    from scipy.stats import linregress

    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)

    v_t, a_t = y_true[:,0], y_true[:,1]
    v_p, a_p = y_pred[:,0], y_pred[:,1]
    sv = linregress(v_t, v_p); sa = linregress(a_t, a_p)
    return {
        "valence": {"slope": sv.slope, "intercept": sv.intercept, "r": sv.rvalue,
                    "std_true": float(np.std(v_t)), "std_pred": float(np.std(v_p))},
        "arousal": {"slope": sa.slope, "intercept": sa.intercept, "r": sa.rvalue,
                    "std_true": float(np.std(a_t)), "std_pred": float(np.std(a_p))},
    }

# End-to-end runner

@dataclass
class RunOutputs:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray
    ridge_val: dict
    ridge_test: dict
    Y_val_pred: np.ndarray
    Y_test_pred: np.ndarray
    mlp_val: dict
    mlp_test: dict
    train_users: set
    val_users: set
    test_users: set
    mlp_model: "torch.nn.Module"
    model_name: str
    max_len: int
    use_cls: bool
    

def run_llm_pipeline(posts_csv_path: str) -> RunOutputs:
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

    # load & split
    df = load_posts_csv(posts_csv_path)
    train_df, val_df, test_df = split_by_user(df, test_frac=0.15, val_frac=0.10, random_state=42)

    # encode
    X_train = encode_texts(train_df["clean_text"].tolist(), batch_size=BATCH_ENC, use_cls=USE_CLS)
    X_val   = encode_texts(val_df["clean_text"].tolist(),   batch_size=BATCH_ENC, use_cls=USE_CLS)
    X_test  = encode_texts(test_df["clean_text"].tolist(),  batch_size=BATCH_ENC, use_cls=USE_CLS)
    Y_train = train_df[["valence_w","arousal_w"]].to_numpy(np.float32)
    Y_val   = val_df  [["valence_w","arousal_w"]].to_numpy(np.float32)
    Y_test  = test_df [["valence_w","arousal_w"]].to_numpy(np.float32)
    train_users = set(train_df["user_id"].dropna().unique().tolist())
    val_users   = set(val_df  ["user_id"].dropna().unique().tolist())
    test_users  = set(test_df ["user_id"].dropna().unique().tolist())

    # ridge baseline
    ridge_val, ridge_test = ridge_baseline(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    print("Ridge VAL :", ridge_val)
    print("Ridge TEST:", ridge_test)

    # MLP 
    cfg = MLPConfig(h=512, p=0.05, lr=1e-3, wd=0.0, loss="huber",
                    epochs=12, batch=256, patience=3, seed=SEED)
    mlp_model, Y_val_pred, Y_test_pred = train_mlp_return_preds(
    X_train, Y_train, X_val, Y_val, X_test, cfg
)

    mlp_val  = metrics_dict(Y_val,  Y_val_pred)
    mlp_test = metrics_dict(Y_test, Y_test_pred)
    print("MLP VAL  :", mlp_val)
    print("MLP TEST :", mlp_test)

    # quick calibration print
    cal = calibration(Y_test, Y_test_pred)
    print("Calibration (TEST):", cal)

    return RunOutputs(train_df, val_df, test_df,
                      X_train, X_val, X_test,
                      Y_train, Y_val, Y_test,
                      ridge_val, ridge_test,
                      Y_val_pred, Y_test_pred,
                      mlp_val, mlp_test,
                      train_users, val_users, test_users,
                      mlp_model,
                      model_name=MODEL_NAME,
                      max_len=MAX_LEN,
                      use_cls=USE_CLS,
                      )

# per-post predictions for later LSTM
@torch.no_grad()
def predict_posts(
    df_posts: pd.DataFrame,
    mlp_model: nn.Module,
    batch_size: int = BATCH_ENC,
    use_cls: bool = USE_CLS,
) -> pd.DataFrame:
    """
    Encode texts with the same encoder/pooling used in training,
    then run the trained MLP head to get v_hat / a_hat / r_hat.
    """
    X_all = encode_texts(
        df_posts["clean_text"].astype(str).tolist(),
        batch_size=batch_size,
        use_cls=use_cls,
    )  

    preds = (
        mlp_model.to(DEVICE).eval()(
            torch.from_numpy(X_all).float().to(DEVICE)
        )
        .cpu()
        .numpy()
    )
    out = df_posts.copy()
    out["v_hat"] = preds[:, 0]
    out["a_hat"] = preds[:, 1]
    out["r_hat"] = np.sqrt(out["v_hat"]**2 + out["a_hat"]**2)
    return out



if __name__ == "__main__":
    
    CSV_PATH = "./data/circumplex_post_level.csv"
    if not os.path.exists(CSV_PATH):
        print(f"Update CSV_PATH in __main__: {CSV_PATH} not found.")
    else:
        _ = run_llm_pipeline(CSV_PATH)
