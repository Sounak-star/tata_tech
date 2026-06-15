"""
train_xgboost.py  -  Fatigue / Drowsiness Binary Classifier
           v4: alert vs at-risk (collapsed 3-class -> binary)
=============================================================================

The 3-class split (alert / low-vigilance / drowsy) kept failing on low-
vigilance because the middle class is ambiguous from window features alone.
This version collapses to a clean binary decision:

    drowsiness_level 0  -> 0  ALERT
    drowsiness_level 1  -> 1  AT-RISK
    drowsiness_level 2  -> 1  AT-RISK

Benefits:
  - Single probability output P(at-risk) is easier to threshold.
  - No ambiguous middle class to confuse the model.
  - Severity grading (OK / MILD / STRONG) is done post-hoc from P(at-risk),
    giving a smooth graduated alert instead of hard class boundaries.

Keeps all machinery from v2/v3:
  - Per-subject held-out calibration slice (honest baseline)
  - GroupKFold(n_splits=5) by subject_id
  - SMOTE on the training fold ONLY
  - No feature scaling anywhere (raw values for SHAP)
  - CPU only  (no device="cuda")
  - Temporal smoothing within (subject, video) clips
  - SHAP explain_prediction with per-subject baseline dict

Produces:
  fatigue_binary_model.json    - XGBoost booster (portable JSON)
  fatigue_binary_model_clf.pkl - full sklearn wrapper (joblib)
  feature_names.json           - exact 12-column inference order
  subject_baselines.json       - per-subject alert baselines
"""

# -- stdlib ------------------------------------------------------------------
import json
import warnings
warnings.filterwarnings("ignore")
import joblib

# -- third-party -------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap


# ============================================================
# CONFIG  -  all tunables in one place
# ============================================================

# -- Paths -------------------------------------------------------------------
DATA_PATH           = "raw_features_full.csv"
MODEL_SAVE_PATH     = "fatigue_binary_model.json"
FEAT_SAVE_PATH      = "feature_names.json"
BASELINES_SAVE_PATH = "subject_baselines.json"

# -- Column roles ------------------------------------------------------------
TARGET_COL        = "drowsiness_level"   # original 3-class column
BINARY_TARGET_COL = "at_risk"            # collapsed binary target (added by collapse_labels)
GROUP_COL         = "subject_id"
VIDEO_COL         = "video_name"
WINDOW_COL        = "window_idx"
QUALITY_COL       = "low_quality"

# -- Exactly 12 model features (canonical order for inference) ---------------
FEATURES = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "blink_rate",
    "mar_mean", "mar_max",
    "is_yawn",
    "pitch_mean", "yaw_mean", "roll_mean",
    "pitch_std",
]

# -- Physical-sanity clipping (applied BEFORE normalisation) -----------------
CLIP_BOUNDS = {
    "ear_mean":   (0.0,  0.5),
    "ear_min":    (0.0,  0.5),
    "mar_mean":   (0.0,  1.5),
    "mar_max":    (0.0,  1.5),
    "blink_rate": (0.0, 60.0),
    "pitch_mean": (-90.0, 90.0),
    "yaw_mean":   (-90.0, 90.0),
    "roll_mean":  (-90.0, 90.0),
    "pitch_std":  (0.0,  90.0),
}

# -- XGBoost hyper-parameters (binary, CPU only) -----------------------------
XGB_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "aucpr",
    tree_method      = "hist",        # CPU  -  no device="cuda"
    max_depth        = 5,
    n_estimators     = 300,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = 0,
)

N_SPLITS = 5

# Calibration slice: first N alert windows per subject (~ 30 seconds)
N_CALIB_WINDOWS = 30

# Decision threshold (< 0.5 to favour catching at-risk operators)
AT_RISK_THRESHOLD = 0.40

# Threshold sweep values
THRESHOLD_SWEEP = [0.30, 0.40, 0.50, 0.60]

# Temporal smoothing window (windows)
SMOOTH_WINDOW = 30

# Graduated-alert severity tiers (applied to smoothed P(at-risk))
SEVERITY_OK     = 0.40   # below this -> OK
SEVERITY_MILD   = 0.70   # between OK and MILD -> MILD warning; >= MILD -> STRONG

MIN_CALIB_WINDOWS = 5
EPSILON           = 1e-6

CLASS_NAMES_BIN = ["Alert", "At-risk"]


# ============================================================
# 1.  LOAD & CLEAN
# ============================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Read raw_features_full.csv.  Drop low-quality rows and NaNs, clip to
    physically-sane bounds.  NO scaling - raw values kept for SHAP.
    """
    print(f"[load] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"[load] Raw shape  : {df.shape}")

    required = set(FEATURES + [TARGET_COL, GROUP_COL, QUALITY_COL,
                                VIDEO_COL, WINDOW_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    before = len(df)
    df = df[df[QUALITY_COL] != 1].copy()
    print(f"[clean] Dropped {before - len(df):,} low-quality rows -> {len(df):,} remain")

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET_COL]).copy()
    print(f"[clean] Dropped {before - len(df):,} NaN rows         -> {len(df):,} remain")

    for col, (lo, hi) in CLIP_BOUNDS.items():
        df[col] = df[col].clip(lo, hi)

    print(f"[clean] Final shape : {df.shape}  |  Subjects: {df[GROUP_COL].nunique()}")
    return df


# ============================================================
# 2.  LABEL COLLAPSE  (3-class -> binary)
# ============================================================

def collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary at_risk column:
        drowsiness_level 0  ->  0  (ALERT)
        drowsiness_level 1  ->  1  (AT-RISK)
        drowsiness_level 2  ->  1  (AT-RISK)

    Original TARGET_COL is kept for reference.
    """
    df = df.copy()
    df[BINARY_TARGET_COL] = (df[TARGET_COL] >= 1).astype(int)

    n_alert   = (df[BINARY_TARGET_COL] == 0).sum()
    n_at_risk = (df[BINARY_TARGET_COL] == 1).sum()
    total     = len(df)
    print(f"\n[collapse] Binary class distribution:")
    print(f"   Alert  (0) : {n_alert:,}   ({n_alert/total*100:.1f}%)")
    print(f"   At-risk(1) : {n_at_risk:,}  ({n_at_risk/total*100:.1f}%)")
    ratio = n_at_risk / max(n_alert, 1)
    print(f"   Ratio at-risk:alert = {ratio:.2f}:1\n")
    return df


# ============================================================
# 3.  CALIBRATION SLICE  (held-out; honest baseline)
# ============================================================

def split_calibration_windows(df: pd.DataFrame):
    """
    For each subject reserve first N_CALIB_WINDOWS ALERT (original label=0)
    windows as the calibration slice.  These are used ONLY to compute the
    subject's baseline mean/std and are EXCLUDED from both training and test.
    This mirrors the live 30-second calibration phase exactly.
    """
    calib_idx = []
    for subj_id in df[GROUP_COL].unique():
        subj_alert = df[
            (df[GROUP_COL] == subj_id) & (df[TARGET_COL] == 0)
        ].sort_values([VIDEO_COL, WINDOW_COL])
        n_take = min(N_CALIB_WINDOWS, len(subj_alert))
        if n_take < MIN_CALIB_WINDOWS:
            print(f"  [warn] {subj_id}: only {n_take} alert windows for calibration")
        calib_idx.extend(subj_alert.index[:n_take].tolist())

    calib_df   = df.loc[calib_idx].copy()
    scoring_df = df.drop(index=calib_idx).copy()

    print(f"[calib] Reserved {len(calib_df):,} calibration windows "
          f"(first {N_CALIB_WINDOWS} alert windows/subject)")
    print(f"[calib] Scoring dataset : {len(scoring_df):,} windows\n")
    return calib_df, scoring_df


# ============================================================
# 4.  PER-SUBJECT BASELINE
# ============================================================

def compute_subject_baselines(calib_df: pd.DataFrame) -> dict:
    """
    Compute per-feature mean and std from the held-out calibration slice
    (alert windows only) for each subject.

    Returns: {subject_id: {feature: {"mean": float, "std": float}}}
    """
    baselines = {}
    for subj_id in calib_df[GROUP_COL].unique():
        subj_df = calib_df[calib_df[GROUP_COL] == subj_id]
        subj_bl = {}
        for feat in FEATURES:
            mean = float(subj_df[feat].mean())
            std  = float(subj_df[feat].std())
            if std < EPSILON or np.isnan(std):
                std = 1.0
            subj_bl[feat] = {"mean": mean, "std": std}
        baselines[subj_id] = subj_bl
    return baselines


def apply_baselines(df: pd.DataFrame, baselines: dict) -> np.ndarray:
    """
    Z-score each row relative to that subject's calibration baseline.
    Returns numpy array (n_rows, n_features).  Original df NOT modified.
    """
    result = df[FEATURES].copy().astype(float)
    for subj_id in df[GROUP_COL].unique():
        if subj_id not in baselines:
            continue
        mask    = df[GROUP_COL] == subj_id
        subj_bl = baselines[subj_id]
        for feat in FEATURES:
            m = subj_bl[feat]["mean"]
            s = subj_bl[feat]["std"]
            result.loc[mask, feat] = (result.loc[mask, feat] - m) / (s + EPSILON)
    return result.values


# ============================================================
# 5.  THRESHOLD & SEVERITY
# ============================================================

def apply_threshold(p_at_risk: np.ndarray,
                    threshold: float = AT_RISK_THRESHOLD) -> np.ndarray:
    """Predict 1 (at-risk) if P(at-risk) >= threshold, else 0 (alert)."""
    return (p_at_risk >= threshold).astype(int)


def classify_severity(p_at_risk: float,
                      ok_thresh: float   = SEVERITY_OK,
                      mild_thresh: float = SEVERITY_MILD) -> str:
    """
    Map smoothed P(at-risk) to a graduated severity tier.
    Used by the live UI to select the appropriate alert level.

    P < ok_thresh           -> 'OK'       (no alert)
    ok_thresh <= P < mild   -> 'MILD'     ("take a break soon")
    P >= mild_thresh        -> 'STRONG'   (alert driver immediately)
    """
    if p_at_risk >= mild_thresh:
        return "STRONG"
    elif p_at_risk >= ok_thresh:
        return "MILD"
    return "OK"


# ============================================================
# 6.  TEMPORAL SMOOTHING
# ============================================================

def smooth_p_at_risk(df_meta: pd.DataFrame,
                     p_at_risk: np.ndarray,
                     window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    Rolling-mean smooth of P(at-risk) WITHIN each (subject_id, video_name)
    clip, sorted by window_idx.  Never crosses clip or subject boundaries.

    df_meta   : DataFrame with GROUP_COL, VIDEO_COL, WINDOW_COL
    p_at_risk : 1-D array of per-window probabilities, aligned with df_meta
    Returns   : smoothed 1-D array, same length
    """
    p64 = p_at_risk.astype(np.float64)

    work = pd.DataFrame({
        "_pos":    np.arange(len(df_meta)),
        GROUP_COL: df_meta[GROUP_COL].values,
        VIDEO_COL: df_meta[VIDEO_COL].values,
        WINDOW_COL: df_meta[WINDOW_COL].values,
        "p":       p64,
    })

    pieces = []
    for (_, _), grp in work.groupby([GROUP_COL, VIDEO_COL], sort=False):
        grp_sorted = grp.sort_values(WINDOW_COL).copy()
        grp_sorted["p"] = (
            grp_sorted["p"].rolling(window=window, min_periods=1).mean()
        )
        pieces.append(grp_sorted)

    smoothed = pd.concat(pieces).sort_values("_pos")
    return smoothed["p"].values


# ============================================================
# 7.  THRESHOLD SWEEP
# ============================================================

def threshold_sweep(all_p: np.ndarray, all_y: np.ndarray) -> None:
    """
    Sweep AT_RISK_THRESHOLD over THRESHOLD_SWEEP values.
    Print a table: at-risk recall | alert recall | precision |
                   false-alarm rate | F1 | accuracy.
    False-alarm rate = fraction of truly-alert windows predicted as at-risk.
    """
    print("\n" + "=" * 74)
    print(" THRESHOLD SWEEP  (aggregated held-out data)")
    print("=" * 74)
    hdr = (f"  {'Threshold':>10s}  {'AtRiskRec':>10s}  {'AlertRec':>9s}  "
           f"{'Precision':>10s}  {'F1':>7s}  {'FalseAlarm%':>12s}  {'Accuracy':>9s}")
    print(hdr)
    print("  " + "-" * 70)

    alert_mask = all_y == 0
    for t in THRESHOLD_SWEEP:
        y_pred = apply_threshold(all_p, threshold=t)
        atr_rec = recall_score(all_y, y_pred, pos_label=1, zero_division=0)
        ale_rec = recall_score(all_y, y_pred, pos_label=0, zero_division=0)
        prec    = precision_score(all_y, y_pred, pos_label=1, zero_division=0)
        f1      = f1_score(all_y, y_pred, pos_label=1, zero_division=0)
        fa_rate = ((y_pred == 1) & alert_mask).sum() / max(alert_mask.sum(), 1)
        acc     = accuracy_score(all_y, y_pred)
        print(f"  {t:>10.2f}  {atr_rec:>10.4f}  {ale_rec:>9.4f}  "
              f"{prec:>10.4f}  {f1:>7.4f}  {fa_rate*100:>11.1f}%  {acc:>9.4f}")
    print()


# ============================================================
# 8.  SUBJECT-GROUPED CROSS-VALIDATION
# ============================================================

def run_cv(scoring_df: pd.DataFrame, all_baselines: dict) -> dict:
    """
    5-fold GroupKFold CV on scoring_df (calibration windows excluded).
    Baselines come from the held-out calibration slice.

    Per fold:
      - Normalise with calibration baselines (no leakage)
      - SMOTE on training data only
      - Fit binary XGBoost
      - Evaluate per-window and after temporal smoothing
    Aggregates all held-out probabilities for the threshold sweep.
    """
    y      = scoring_df[BINARY_TARGET_COL].values.astype(int)
    groups = scoring_df[GROUP_COL].values

    gkf   = GroupKFold(n_splits=N_SPLITS)
    smote = SMOTE(random_state=42)

    # Per-window accumulators
    accs_pw = []; recs_atr_pw = []; recs_ale_pw = []
    precs_pw = []; f1s_pw = []
    aucs_roc_pw = []; aucs_pr_pw = []
    cm_pw = np.zeros((2, 2), dtype=int)

    # Smoothed accumulators
    accs_sm = []; recs_atr_sm = []; recs_ale_sm = []
    precs_sm = []; f1s_sm = []
    aucs_roc_sm = []; aucs_pr_sm = []
    cm_sm = np.zeros((2, 2), dtype=int)

    # For threshold sweep
    all_p_list = []
    all_y_list = []

    print("=" * 62)
    print(" Subject-Grouped 5-Fold CV  (binary: alert vs at-risk)")
    print("=" * 62)

    for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(scoring_df, y, groups=groups), start=1):

        train_df = scoring_df.iloc[train_idx].copy()
        test_df  = scoring_df.iloc[test_idx].copy()
        test_subjects = np.unique(groups[test_idx])

        print(f"\n-- Fold {fold_idx}  (test: {test_subjects}) --")
        print(f"   Train: {len(train_df):,}   Test: {len(test_df):,}")

        # Normalise using calibration baselines
        X_tr = apply_baselines(train_df, all_baselines)
        X_te = apply_baselines(test_df,  all_baselines)
        y_tr = train_df[BINARY_TARGET_COL].values.astype(int)
        y_te = test_df[BINARY_TARGET_COL].values.astype(int)

        # SMOTE on training only
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        print(f"   After SMOTE: {len(X_tr_res):,} (was {len(X_tr):,})")

        # Train binary XGBoost
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr_res, y_tr_res, verbose=False)

        # P(at-risk) on held-out test
        p_at_risk = model.predict_proba(X_te)[:, 1]

        # ---- Per-window metrics -------------------------------------------
        y_pred_pw = apply_threshold(p_at_risk)
        _accumulate(y_te, y_pred_pw, p_at_risk,
                    accs_pw, recs_atr_pw, recs_ale_pw,
                    precs_pw, f1s_pw, aucs_roc_pw, aucs_pr_pw, cm_pw)
        print(f"   Per-window : acc={accs_pw[-1]:.4f}  "
              f"at-risk-rec={recs_atr_pw[-1]:.4f}  "
              f"alert-rec={recs_ale_pw[-1]:.4f}  "
              f"ROC-AUC={aucs_roc_pw[-1]:.4f}")

        # ---- Temporal smoothing -------------------------------------------
        p_sm   = smooth_p_at_risk(test_df, p_at_risk)
        y_pred_sm = apply_threshold(p_sm)
        _accumulate(y_te, y_pred_sm, p_sm,
                    accs_sm, recs_atr_sm, recs_ale_sm,
                    precs_sm, f1s_sm, aucs_roc_sm, aucs_pr_sm, cm_sm)
        print(f"   Smoothed   : acc={accs_sm[-1]:.4f}  "
              f"at-risk-rec={recs_atr_sm[-1]:.4f}  "
              f"alert-rec={recs_ale_sm[-1]:.4f}  "
              f"ROC-AUC={aucs_roc_sm[-1]:.4f}")

        all_p_list.append(p_at_risk)
        all_y_list.append(y_te)

    # ---- Print CV summary -------------------------------------------------
    _print_cv_summary(
        accs_pw,  recs_atr_pw, recs_ale_pw, precs_pw,
        f1s_pw,   aucs_roc_pw, aucs_pr_pw,  cm_pw,
        accs_sm,  recs_atr_sm, recs_ale_sm, precs_sm,
        f1s_sm,   aucs_roc_sm, aucs_pr_sm,  cm_sm,
    )

    # ---- Threshold sweep on aggregated held-out probabilities --------------
    all_p_agg = np.concatenate(all_p_list)
    all_y_agg = np.concatenate(all_y_list)
    threshold_sweep(all_p_agg, all_y_agg)

    # Final operating-point summary
    y_pred_chosen = apply_threshold(all_p_agg, AT_RISK_THRESHOLD)
    chosen_rec = recall_score(all_y_agg, y_pred_chosen, pos_label=1, zero_division=0)
    chosen_fa  = (
        ((y_pred_chosen == 1) & (all_y_agg == 0)).sum() /
        max((all_y_agg == 0).sum(), 1)
    )
    print(f"  Chosen threshold = {AT_RISK_THRESHOLD:.2f}:")
    print(f"    At-risk recall  = {chosen_rec:.4f}  "
          f"({chosen_rec*100:.1f}% of at-risk windows caught)")
    print(f"    False-alarm rate= {chosen_fa:.4f}  "
          f"({chosen_fa*100:.1f}% of alert windows flagged)\n")

    return {
        "acc_pw":         np.mean(accs_pw),
        "rec_atr_pw":     np.mean(recs_atr_pw),
        "rec_ale_pw":     np.mean(recs_ale_pw),
        "prec_pw":        np.mean(precs_pw),
        "f1_pw":          np.mean(f1s_pw),
        "roc_auc_pw":     np.mean(aucs_roc_pw),
        "pr_auc_pw":      np.mean(aucs_pr_pw),
        "acc_sm":         np.mean(accs_sm),
        "rec_atr_sm":     np.mean(recs_atr_sm),
        "rec_ale_sm":     np.mean(recs_ale_sm),
        "prec_sm":        np.mean(precs_sm),
        "f1_sm":          np.mean(f1s_sm),
        "roc_auc_sm":     np.mean(aucs_roc_sm),
        "pr_auc_sm":      np.mean(aucs_pr_sm),
    }


def _accumulate(y_te, y_pred, p_score,
                accs, recs_atr, recs_ale, precs, f1s,
                aucs_roc, aucs_pr, cm):
    """Helper: compute binary metrics for one fold and append to lists."""
    accs.append(accuracy_score(y_te, y_pred))
    recs_atr.append(recall_score(y_te, y_pred, pos_label=1, zero_division=0))
    recs_ale.append(recall_score(y_te, y_pred, pos_label=0, zero_division=0))
    precs.append(precision_score(y_te, y_pred, pos_label=1, zero_division=0))
    f1s.append(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
    try:
        aucs_roc.append(roc_auc_score(y_te, p_score))
        aucs_pr.append(average_precision_score(y_te, p_score))
    except ValueError:
        aucs_roc.append(float("nan"))
        aucs_pr.append(float("nan"))
    cm += confusion_matrix(y_te, y_pred, labels=[0, 1])


def _print_cv_summary(
    accs_pw, recs_atr_pw, recs_ale_pw, precs_pw,
    f1s_pw, aucs_roc_pw, aucs_pr_pw, cm_pw,
    accs_sm, recs_atr_sm, recs_ale_sm, precs_sm,
    f1s_sm, aucs_roc_sm, aucs_pr_sm, cm_sm,
):
    print("\n" + "=" * 62)
    print(" CV SUMMARY  (binary: alert vs at-risk)")
    print("=" * 62)

    metrics = [
        ("Accuracy",        accs_pw,     accs_sm),
        ("At-risk recall !", recs_atr_pw, recs_atr_sm),
        ("Alert recall",    recs_ale_pw, recs_ale_sm),
        ("Precision (atr)", precs_pw,    precs_sm),
        ("F1 (at-risk)",    f1s_pw,      f1s_sm),
        ("ROC-AUC",         aucs_roc_pw, aucs_roc_sm),
        ("PR-AUC",          aucs_pr_pw,  aucs_pr_sm),
    ]

    print(f"\n  {'Metric':22s}  {'Per-window (mean+-std)':>24s}  "
          f"{'Smoothed (mean+-std)':>22s}")
    print("  " + "-" * 72)
    for name, pw, sm in metrics:
        flag = " <-- SAFETY" if "!" in name else ""
        clean = name.replace("!", "")
        print(f"  {clean:22s}  "
              f"{np.nanmean(pw):>7.4f} +- {np.nanstd(pw):.4f}           "
              f"{np.nanmean(sm):>7.4f} +- {np.nanstd(sm):.4f}{flag}")

    print(f"\n  Per-window confusion matrix (rows=true, cols=pred):")
    _print_cm2(cm_pw)
    print(f"\n  Smoothed   confusion matrix:")
    _print_cm2(cm_sm)


def _print_cm2(cm: np.ndarray) -> None:
    print(f"  {'':>12s}  {'Pred Alert':>12s}  {'Pred At-risk':>12s}")
    for i, row_name in enumerate(CLASS_NAMES_BIN):
        print(f"  {'True '+row_name:>12s}  {cm[i,0]:>12,}  {cm[i,1]:>12,}")


# ============================================================
# 9.  FINAL MODEL
# ============================================================

def train_final(scoring_df: pd.DataFrame, all_baselines: dict):
    """
    Train final binary model on all scoring data (calibration excluded).
    SMOTE on the full set is fine here (no held-out test remains).
    Saves model, feature list, baselines.
    """
    print("\n" + "=" * 62)
    print(" TRAINING FINAL BINARY MODEL (all scoring data)")
    print("=" * 62)

    X = apply_baselines(scoring_df, all_baselines)
    y = scoring_df[BINARY_TARGET_COL].values.astype(int)

    smote       = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  After SMOTE: {len(X_res):,} rows  (was {len(X):,})")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_res, y_res, verbose=False)
    print("  Training complete.")

    # Save booster as JSON (xgboost 3.x safe)
    model.get_booster().save_model(MODEL_SAVE_PATH)
    print(f"  Booster saved        -> {MODEL_SAVE_PATH}")

    pkl_path = MODEL_SAVE_PATH.replace(".json", "_clf.pkl")
    joblib.dump(model, pkl_path)
    print(f"  Classifier saved     -> {pkl_path}")

    with open(FEAT_SAVE_PATH, "w") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"  Feature list saved   -> {FEAT_SAVE_PATH}")

    with open(BASELINES_SAVE_PATH, "w") as f:
        json.dump(all_baselines, f, indent=2)
    print(f"  Baselines saved      -> {BASELINES_SAVE_PATH}")

    fi  = dict(zip(FEATURES, model.feature_importances_))
    top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n  Top-5 feature importances (gain):")
    for feat, imp in top:
        print(f"    {feat:<20s}  {imp:.4f}")

    return model


# ============================================================
# 10. SHAP / EXPLAIN FUNCTION
# ============================================================

_explainer:   shap.TreeExplainer | None = None
_final_model: xgb.XGBClassifier | None = None


def build_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Fit and cache a SHAP TreeExplainer on the final binary model."""
    global _explainer, _final_model
    print("\n[SHAP] Fitting TreeExplainer ...")
    _explainer   = shap.TreeExplainer(model)
    _final_model = model
    print("[SHAP] Explainer ready.")
    return _explainer


def explain_prediction(
    feature_row,
    subject_baseline: dict | None = None,
) -> dict:
    """
    Explain one binary prediction with SHAP.

    Parameters
    ----------
    feature_row      : dict / pd.Series / np.ndarray  with the 12 RAW features.
    subject_baseline : {feature: {"mean": float, "std": float}}
                       from the 30-sec live calibration.  If None, raw values
                       are passed to the model (reduced accuracy; warns).

    Returns
    -------
    {
      "alert":        bool    (True = ALERT, False = AT-RISK),
      "p_at_risk":    float   (smoothed P(at-risk) for graduated alert),
      "severity":     str     ("OK" / "MILD" / "STRONG"),
      "reasons":      list of (feature, raw_value, direction) tuples
                      where direction is "high" or "low" relative to baseline.
                      Raw values are used (not z-scores) for Reason Cards.
    }

    Live UI usage:
      result = explain_prediction(window_features, calibration_baseline)
      if result["severity"] == "STRONG":
          show_alert("Driver may be drowsy!")
          for feat, val, dir in result["reasons"]:
              show_reason_card(feat, val, dir)
    """
    if _explainer is None or _final_model is None:
        raise RuntimeError("Call build_explainer(model) before explain_prediction().")

    # Normalise input
    if isinstance(feature_row, dict):
        raw_dict = {f: feature_row[f] for f in FEATURES}
    elif isinstance(feature_row, pd.Series):
        raw_dict = feature_row[FEATURES].to_dict()
    else:
        raw_dict = dict(zip(FEATURES, feature_row))

    raw_values = np.array([raw_dict[f] for f in FEATURES], dtype=float)

    if subject_baseline is not None:
        norm_values = np.array([
            (raw_dict[f] - subject_baseline[f]["mean"]) /
            (subject_baseline[f]["std"] + EPSILON)
            for f in FEATURES
        ], dtype=float)
    else:
        print("[warn] No subject_baseline; using raw values (reduced accuracy).")
        norm_values = raw_values.copy()

    row_arr   = norm_values.reshape(1, -1)
    p_at_risk = float(_final_model.predict_proba(row_arr)[0, 1])
    is_alert  = p_at_risk < AT_RISK_THRESHOLD
    severity  = classify_severity(p_at_risk)

    # SHAP values for binary:logistic
    # shap 0.48+ may return 2-D (n_samples, n_features) for binary,
    # or a list [shap_class0, shap_class1], or 3-D (n,f,2).
    raw_sv = _explainer.shap_values(row_arr)

    if isinstance(raw_sv, list):
        # List form: take index 1 (at-risk contributions)
        sv = raw_sv[1][0] if raw_sv[1].ndim == 2 else raw_sv[1]
    elif isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
        # (n_samples, n_features, 2)  ->  class-1 slice
        sv = raw_sv[0, :, 1]
    else:
        # 2-D (n_samples, n_features) for binary - already class-1 log-odds
        sv = raw_sv[0]

    # Top 3 features by |SHAP contribution|
    top_indices = np.argsort(np.abs(sv))[::-1][:3]
    reasons = [
        (
            FEATURES[i],
            round(float(raw_values[i]), 4),
            "high" if sv[i] > 0 else "low",
        )
        for i in top_indices
    ]

    return {
        "alert":     is_alert,
        "p_at_risk": round(p_at_risk, 4),
        "severity":  severity,
        "reasons":   reasons,
    }


# ============================================================
# 11. MAIN
# ============================================================

def main():
    # 1. Load and clean
    df = load_and_clean(DATA_PATH)

    # 2. Collapse to binary labels
    df = collapse_labels(df)

    # 3. Split off calibration windows (honest held-out baseline)
    calib_df, scoring_df = split_calibration_windows(df)

    # 4. Compute per-subject baselines from calibration slice ONLY
    all_baselines = compute_subject_baselines(calib_df)
    print(f"[baselines] Computed for {len(all_baselines)} subjects.")

    # 5. Leak-free CV
    cv_metrics = run_cv(scoring_df, all_baselines)

    # 6. Train final model on all scoring data
    final_model = train_final(scoring_df, all_baselines)

    # 7. Build SHAP explainer
    build_explainer(final_model)

    # 8. Demo explain_prediction() on a drowsy sample
    print("\n" + "=" * 62)
    print(" DEMO: explain_prediction()")
    print("=" * 62)

    sample_baseline = {
        "ear_mean":   {"mean": 0.29, "std": 0.03},
        "ear_min":    {"mean": 0.20, "std": 0.04},
        "ear_std":    {"mean": 0.03, "std": 0.01},
        "perclos":    {"mean": 0.05, "std": 0.02},
        "blink_rate": {"mean": 18.0, "std": 4.0},
        "mar_mean":   {"mean": 0.10, "std": 0.05},
        "mar_max":    {"mean": 0.25, "std": 0.08},
        "is_yawn":    {"mean": 0.02, "std": 0.14},
        "pitch_mean": {"mean": -2.0, "std": 5.0},
        "yaw_mean":   {"mean":  1.0, "std": 8.0},
        "roll_mean":  {"mean":  0.5, "std": 3.0},
        "pitch_std":  {"mean":  3.0, "std": 2.0},
    }
    sample_raw = {
        "ear_mean": 0.17, "ear_min": 0.10, "ear_std": 0.04,
        "perclos": 0.68,  "blink_rate": 12.0,
        "mar_mean": 0.62, "mar_max": 1.10, "is_yawn": 1.0,
        "pitch_mean": -8.0, "yaw_mean": 3.0,
        "roll_mean": 2.0, "pitch_std": 5.0,
    }

    result = explain_prediction(sample_raw, subject_baseline=sample_baseline)

    status = "ALERT" if result["alert"] else "AT-RISK"
    print(f"\n  Status    : {status}")
    print(f"  P(at-risk): {result['p_at_risk']:.4f}")
    print(f"  Severity  : {result['severity']}")
    print("\n  Reason Cards:")

    TEMPLATES = {
        "perclos":    lambda v, d: f"Eyes closed {v*100:.0f}% of the time",
        "ear_mean":   lambda v, d: f"Eye openness {'very low' if d=='low' else 'normal'} ({v:.2f})",
        "ear_min":    lambda v, d: f"Minimum eye openness {v:.2f}",
        "ear_std":    lambda v, d: f"Eye openness variability {v:.3f}",
        "blink_rate": lambda v, d: f"Blink rate {'high' if d=='high' else 'low'} ({v:.0f}/min)",
        "is_yawn":    lambda v, d: f"Yawn {'detected' if v > 0.5 else 'not detected'}",
        "mar_mean":   lambda v, d: f"Mouth openness avg {v:.2f}",
        "mar_max":    lambda v, d: f"Mouth openness peak {v:.2f}",
        "pitch_mean": lambda v, d: f"Head tilt {v:+.1f} deg (pitch)",
        "yaw_mean":   lambda v, d: f"Head turn {v:+.1f} deg (yaw)",
        "roll_mean":  lambda v, d: f"Head roll {v:+.1f} deg",
        "pitch_std":  lambda v, d: f"Head variability {v:.1f} deg",
    }
    for feat, val, direction in result["reasons"]:
        text = TEMPLATES[feat](val, direction) if feat in TEMPLATES else f"{feat}={val}"
        print(f"    * {text}")

    print("\n[DONE] Artefacts saved:")
    print(f"  {MODEL_SAVE_PATH}")
    print(f"  {MODEL_SAVE_PATH.replace('.json', '_clf.pkl')}")
    print(f"  {FEAT_SAVE_PATH}")
    print(f"  {BASELINES_SAVE_PATH}")


if __name__ == "__main__":
    main()
