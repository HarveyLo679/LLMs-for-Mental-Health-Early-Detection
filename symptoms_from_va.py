import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH  = "./data/circumplex_weekly_user.csv"
OUT_PATH = "./data/outputs/symptoms_weekly_user.csv"

# Prototypes in VA space
PROTOS = {
    "anx":   (-0.4,  0.6),   # anxiety: neg V, high A
    "dep":   (-0.7, -0.4),   # depression: neg V, low A 
    "ptsd":  (-0.5,  0.5),   # trauma-like: neg V, high A
    "mania": ( 0.4,  0.7),   # mania-like: pos V, high A
}
# Suicidality has two affective modes
SUI_PROTOS = [(-0.9, +0.2), (-0.9, -0.2)]

SIGMA  = 0.35   # Gaussian 
ALPHA  = 0.6    # weight 
BETA   = 1.0    # steepness for sigmoid on Empath evidence
GAMMA  = 0.15   # small boost by radius magnitude 

THR_GENERIC = 0.50
THR_SUI     = 0.55
THR_LOW  = 0.45

# Empath weekly columns expected in input
EMPATH_COLS = [
 'empath_anxiety_mean',
 'empath_pain_mean',
 'empath_violence_mean',
 'empath_sadness_mean',
 'empath_depression_mean',
 'empath_help_mean',
 'empath_love_mean',
 'empath_suffering_mean',
 'empath_health_mean',
 'empath_contentment_mean',
 'empath_fear_mean',
 'empath_trust_mean',
 'empath_hate_mean',
 'empath_therapy_mean',
 'empath_sleep_mean',
 'empath_anger_mean',
 'empath_shame_mean',
 'empath_optimism_mean',
]

def _zscore_cols(df: pd.DataFrame, cols):
    """Return a copy with z-scored columns"""
    dfz = df.copy()
    for c in cols:
        if c not in dfz.columns:
            dfz[c] = 0.0
        mu = dfz[c].mean()
        sd = dfz[c].std(ddof=0)
        dfz[c] = 0.0 if (sd is None or sd == 0) else (dfz[c] - mu) / sd
    return dfz

def _nz(x):  # nonnegative clamp
    return x if x > 0 else 0.0

def _sim(V, A, muV, muA, s=SIGMA):
    return np.exp(-(((V - muV)**2) / (s**2) + ((A - muA)**2) / (s**2)))

def main():
    Path("./data/outputs").mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(IN_PATH)

    # normalize VA column names if needed
    df = df_orig.rename(columns={
        "valence_z_mean": "valence_w",
        "arousal_z_mean": "arousal_w",
        "radius_z_mean":  "radius_w",
    })
    for c in ["user_id","week","valence_w","arousal_w","radius_w"]:
        if c not in df.columns:
            raise ValueError(f"Missing column in input: {c}")

    df_emz = _zscore_cols(df[["user_id","week"] + [c for c in EMPATH_COLS if c in df.columns]], EMPATH_COLS)
    df_emz = df_emz.rename(columns={c: f"{c}_z" for c in EMPATH_COLS})
    dfj = df.merge(df_emz, on=["user_id","week"], how="left").fillna(0.0)

    # evidence functions
    def E_anx(r):  return _nz(r.get("empath_anxiety_mean_z",0)) + _nz(r.get("empath_fear_mean_z",0))
    def E_dep(r):  return _nz(r.get("empath_sadness_mean_z",0)) + _nz(r.get("empath_depression_mean_z",0)) + _nz(-r.get("empath_sleep_mean_z",0))
    def E_ptsd(r): return _nz(r.get("empath_violence_mean_z",0)) + _nz(r.get("empath_suffering_mean_z",0)) + _nz(r.get("empath_pain_mean_z",0))
    def E_mania(r):return _nz(r.get("empath_optimism_mean_z",0)) + _nz(-r.get("empath_sleep_mean_z",0))
    def E_sui(r):  return (_nz(r.get("empath_sadness_mean_z",0)) + _nz(r.get("empath_depression_mean_z",0)) +
                           _nz(r.get("empath_suffering_mean_z",0)) + _nz(r.get("empath_pain_mean_z",0)) +
                           _nz(r.get("empath_fear_mean_z",0)) + _nz(r.get("empath_shame_mean_z",0)) +
                           _nz(r.get("empath_help_mean_z",0)) + _nz(r.get("empath_therapy_mean_z",0)) +
                           _nz(-r.get("empath_sleep_mean_z",0)))

    evid = {"anx":E_anx,"dep":E_dep,"ptsd":E_ptsd,"mania":E_mania}

    # four single-prototype symptoms
    for k,(muV,muA) in PROTOS.items():
        S = _sim(dfj["valence_w"], dfj["arousal_w"], muV, muA) * (1.0 + GAMMA*dfj["radius_w"])
        E = 1.0/(1.0 + np.exp(-BETA * dfj.apply(evid[k], axis=1)))
        dfj[f"{k}_score"] = ALPHA*S + (1.0-ALPHA)*E

    # suicidality
    S_ag = _sim(dfj["valence_w"], dfj["arousal_w"], *SUI_PROTOS[0])
    S_sh = _sim(dfj["valence_w"], dfj["arousal_w"], *SUI_PROTOS[1])
    S_sui = np.maximum(S_ag, S_sh) * (1.0 + GAMMA*dfj["radius_w"])
    E_sui_vals = 1.0/(1.0 + np.exp(-BETA * dfj.apply(E_sui, axis=1)))
    dfj["sui_score"] = ALPHA*S_sui + (1.0-ALPHA)*E_sui_vals

    # labels
    dfj["anx_label"]   = (dfj["anx_score"]   >= THR_GENERIC).astype(int)
    dfj["dep_label"]   = (dfj["dep_score"]   >= THR_LOW).astype(int)
    dfj["ptsd_label"]  = (dfj["ptsd_score"]  >= THR_GENERIC).astype(int)
    dfj["mania_label"] = (dfj["mania_score"] >= THR_LOW).astype(int)
    dfj["sui_label"]   = (dfj["sui_score"]   >= THR_SUI).astype(int)

    KEEP_EMPATH_Z = True
    new_cols = ["anx_score","dep_score","ptsd_score","mania_score","sui_score",
                "anx_label","dep_label","ptsd_label","mania_label","sui_label"]
    df_out = df.copy()                         # start from original
    df_out[new_cols] = dfj[new_cols].values    # append new targets
    # drop all empath_* columns
    drop_empath = [c for c in df_out.columns if c.startswith("empath_")]
    df_out = df_out.drop(columns=drop_empath)
    
    if KEEP_EMPATH_Z:
        z_cols = [c for c in dfj.columns if c.endswith("_z")]
        df_out = df_out.merge(dfj[["user_id","week"] + z_cols], on=["user_id","week"], how="left")

    df_out.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH} (Empath _z kept: {KEEP_EMPATH_Z})")

if __name__ == "__main__":
    main()
