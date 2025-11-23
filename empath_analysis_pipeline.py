import pandas as pd
import numpy as np
from empath import Empath
from pandas import json_normalize

def extract_empath_features(df, text_column='clean_text', 
                           user_id_column='user_id', 
                           category_column='category',
                           optimize_features=True):
    print("Initializing Empath lexicon...")
    lexicon = Empath()

    required_columns = [user_id_column, "thread_owner_id", "post_id", "comment_id", 
                        "is_post", "timestamp", category_column, text_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns}. Using available columns only.")
        available_columns = [col for col in required_columns if col in df.columns]
        empath_out = df[available_columns].copy()
    else:
        empath_out = df[required_columns].copy()

    print("Applying Empath analysis to text...")
    if text_column not in empath_out.columns:
        print(f"Warning: '{text_column}' not found; skipping Empath analysis for this batch.")
        empath_out["liwc_empath"] = [{} for _ in range(len(empath_out))]
    else:
        empath_out['liwc_empath'] = empath_out[text_column].apply(
            lambda x: lexicon.analyze(x, normalize=True) if isinstance(x, str) and x.strip() else {}
        )

    empath_out['date'] = pd.to_datetime(empath_out['timestamp'], errors='coerce')
    empath_out['week'] = empath_out['date'].dt.to_period('W').apply(lambda r: r.start_time)

    print(f"Empath analysis complete. Sample result: {empath_out['liwc_empath'].iloc[0] if len(empath_out) > 0 else 'No data'}")

    print("Expanding Empath features into separate columns...")
    empath_norm = json_normalize(empath_out['liwc_empath']).reset_index(drop=True)
    empath_norm.columns = [f"empath_{col}" for col in empath_norm.columns]
    empath_out = empath_out.drop(columns=['liwc_empath']).reset_index(drop=True)
    empath_out = pd.concat([empath_out, empath_norm], axis=1)
    empath_cols = [c for c in empath_out.columns if c.startswith("empath_")]
    empath_out[empath_cols] = empath_out[empath_cols].fillna(0.0).astype(float)

    if optimize_features:
        empath_out = optimize_empath_features(
            empath_out,
            category_column,
            have_vad=False  
        )
    empath_weekly = create_weekly_features(empath_out, user_id_column)

    print(f"Feature extraction complete!")
    print(f"Post-level features: {empath_out.shape}")
    print(f"Weekly features: {empath_weekly.shape}")

    return empath_out, empath_weekly


def optimize_empath_features(df, 
                            category_column, 
                            have_vad=False,           # set True only if valence_z/arousal_z/radius exist in empath_out
                            min_prevalence=0.05,      # drop rare features
                            min_std=1e-4,             # drop near-constant features
                            corr_min=0.15,            # relevance cutoff to keep vs VAD
                            cluster_r=0.8,            # treat |r|>=0.8 as redundant
                            max_k = 30,
                            must_have=None,
                            alias_map=None
                            ):

    if must_have is None:
        must_have = [
            "empath_suicide","empath_self_harm","empath_therapy","empath_counseling",
            "empath_medication","empath_sleep","empath_pain","empath_family","empath_friends",
            "empath_isolation","empath_support","empath_work","empath_money",
            "empath_depression","empath_anxiety","empath_sadness","empath_anger","empath_fear",
            "empath_stress","empath_worry","empath_loneliness","empath_shame","empath_guilt"
        ]
        
    if alias_map is None:
        alias_map = {
            "empath_nervousness": "empath_anxiety",
            "empath_timidity": "empath_anxiety",
            "empath_rage": "empath_anger",
            "empath_disappointment": "empath_sadness",
            "empath_hopelessness": "empath_depression",
            "empath_helplessness": "empath_depression",
        }
    
    cols = [col for col in df.columns if col.startswith('empath_')]
    print(f"Starting with {len(cols)} Empath features")

    X = df[cols].copy()

    # Merge aliases
    for src, tgt in alias_map.items():
        if src in X.columns:
            if tgt not in X.columns:
                X = X.rename(columns={src: tgt})
            else:
                X[tgt] = X[tgt].fillna(0) + X[src].fillna(0)
                X = X.drop(columns=[src])
    
    # Prevalence and variance
    prev = (X > 0).mean(axis=0)
    stds = X.std(axis=0)
    keep_mask = (prev >= min_prevalence) & (stds >= min_std)
    X = X.loc[:, keep_mask]
    if X.shape[1] == 0:
        print("No features passed basic filters; returning original empath_* as fallback.")
        X = df[[c for c in df.columns if c.startswith("empath_")]].copy()
    print(f"After prevalence/variance: {X.shape[1]}")
    
    # rank by prevalence then variance
    rank_df = pd.DataFrame({"prev": prev.reindex(X.columns).fillna(0),
                            "var":  stds.reindex(X.columns).fillna(0)})
    ranked = rank_df.sort_values(["prev","var"], ascending=[False,False]).index.tolist()
    
    # greedy redundancy pruning on z-scored values
    Z = X.copy()
    for c in Z.columns:
        s = Z[c]
        sd = s.std(ddof=0)
        if sd > 0:
            Z[c] = (s - s.mean()) / sd
        else:
            Z[c] = 0.0

    kept = []
    for c in ranked:
        if not kept:
            kept.append(c)
            continue
        # check corr with already kept
        corr_ok = True
        for k in kept:
            r = float(np.corrcoef(Z[c].values, Z[k].values)[0,1])
            if np.isnan(r): 
                r = 0.0
            if abs(r) >= cluster_r:
                corr_ok = False
                break
        if corr_ok:
            kept.append(c)
        if len(kept) >= max_k:  # stop early at budget
            break
        
    #  must-have restore 
    present_mh = [c for c in must_have if c in X.columns]
    final = list(dict.fromkeys(kept + present_mh))  # preserve order, add MH
    if len(final) > max_k:
        # keep all must-haves; trim others by their rank order
        mh_set = set(present_mh)
        others = [c for c in final if c not in mh_set]
        final = present_mh + others[:max_k - len(present_mh)]
        final = sorted(set(final))
    print(f"Final optimized feature set: {len(final)}")

    # return with metadata
    meta = ['user_id','timestamp','week','date','post_id','comment_id','thread_owner_id','is_post','category', 'clean_text']
    meta_cols = [c for c in df.columns if c in meta]
    out = pd.concat([df[meta_cols].reset_index(drop=True),
                     X[final].reset_index(drop=True)], axis=1)

    with open("empath_selected_columns.txt","w") as f:
        for c in final:
            f.write(c + "\n")
    print("Saved empath_selected_columns.txt")
    return out
    
def create_weekly_features(empath_out, user_id_column):
    """
    Create weekly aggregated features grouped by user and week
    """
    empath_cols = [col for col in empath_out.columns if col.startswith('empath_')]
    
    empath_weekly = (
        empath_out.groupby([user_id_column, 'week'])[empath_cols]
        .mean()
        .reset_index()
    )
    
    return empath_weekly

def save_empath_features(empath_out, empath_weekly, 
                        output_path=".data/empath_data.csv", 
                        weekly_path=".data/empath_weekly.csv"):
    """
    Save Empath features to CSV files
    """
    empath_out.to_csv(output_path, index=False)
    empath_weekly.to_csv(weekly_path, index=False)
    print(f"Saved empath features to {output_path}")
    print(f"Saved weekly features to {weekly_path}")

def load_empath_features(data_path=".data/empath_data.csv", 
                        weekly_path=".data/empath_weekly.csv"):
    """
    Load previously saved Empath features
    """
    empath_df = pd.read_csv(data_path)
    empath_weekly = pd.read_csv(weekly_path)
    
    print(f"Loaded empath features: {empath_df.shape}")
    print(f"Loaded weekly features: {empath_weekly.shape}")
    
    return empath_df, empath_weekly

def analyze_empath_results(empath_df):
    """
    Analyze and display top Empath features
    """
    # Select only empath columns
    empath_cols = [col for col in empath_df.columns if col.startswith('empath_')]
    top_empath = empath_df[empath_cols].mean().sort_values(ascending=False).head(10)
    
    print("Top 10 Empath Categories:")
    for category, score in top_empath.items():
        clean_name = category.replace('empath_', '')
        print(f"  {clean_name:<20}: {score:.4f}")
    
    return top_empath
