# -*- coding: utf-8 -*-
"""
论文图表 v2.4 — 统一配色 + 图文细节修复
"""
import os, warnings, gc, json
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from math import pi

warnings.filterwarnings("ignore")

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ============================================================
# 0. 全局
# ============================================================
EXCEL_CANDIDATES = [
    Path(os.getenv("QSAR_EXCEL_FILE", "")),
    Path(r"D:\res\6.8\data_with_alkane.xlsx"),
    Path(r"D:\res\5.8\data_with_alkane.xlsx"),
    Path(r"D:\Users\hkbg\Desktop\小论文\data_with_alkane.xlsx"),
    Path(r"D:\Users\hkbg\Desktop\大论文\data_with_alkane.xlsx"),
    Path(r"D:\Users\hkbg\Desktop\新建文件夹 (2)\data_with_alkane.xlsx"),
    Path(r"D:\Users\hkbg\Desktop\data_with_alkane.xlsx"),
    Path(r"/data_with_alkane.xlsx"),
]

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "paper_figures_final"
SHAP_DIR = RESULT_DIR / "SHAP"
AD_DIR = RESULT_DIR / "Applicability_Domain"
MODEL_DIR = RESULT_DIR / "model_artifacts"
CONFORMAL_DIR = RESULT_DIR / "Conformal"
WEB_MODEL_DIR = PROJECT_DIR / "web" / "model"
for d in [RESULT_DIR, SHAP_DIR, AD_DIR, MODEL_DIR, CONFORMAL_DIR, WEB_MODEL_DIR]: d.mkdir(parents=True, exist_ok=True)

def ensure_dirs():
    for folder in [RESULT_DIR, SHAP_DIR, AD_DIR, MODEL_DIR, CONFORMAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

SYSTEMS = {"OH":"logkOH","O3":"logkO3","NO3":"logkNO3"}
OXIDANT_E0 = {"OH":2.80,"O3":2.07,"NO3":2.40}
RANDOM_STATE = 42; TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# ===== 统一配色 (6张图完全一致) =====
C_PRIMARY  = "#1B4F72"    # 深蓝(主色)
C_ACCENT   = "#2E86C1"    # 辅蓝
C_LIGHT    = "#AED6F1"    # 浅蓝
C_ORANGE   = "#E67E22"    # 橙色
C_ORANGE_L = "#FAD7A0"    # 浅橙
C_GREEN    = "#1E8449"    # 绿色
C_GREEN_L  = "#A9DFBF"    # 浅绿(新增)
C_RED      = "#C0392B"    # 红色
C_PURPLE   = "#8E44AD"    # 紫色
C_PURPLE_L = "#E8DAEF"    # 浅紫(新增)
C_GRAY     = "#566573"    # 灰色

# 模型颜色(全局统一)
M_COLORS = {"Ridge": C_ACCENT, "SVR": C_ORANGE, "RF": C_GREEN, "GBDT": C_PURPLE}
M_MARKERS = {"Ridge":"o","SVR":"s","RF":"^","GBDT":"D"}
S_COLORS = {"OH":C_PRIMARY,"O3":C_ORANGE,"NO3":C_GREEN}

CMAP_BLUE = LinearSegmentedColormap.from_list("b",["#FDFEFE","#AED6F1","#1B4F72"])
CMAP_BWOR = LinearSegmentedColormap.from_list("bwor",["#1B4F72","#FDFEFE","#E67E22"])

plt.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif","serif"],
    "font.weight":"bold","mathtext.fontset":"stix","mathtext.default":"bf",
    "axes.labelsize":14,"axes.titlesize":15,"axes.titleweight":"bold","axes.labelweight":"bold",
    "axes.linewidth":1.2,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":10.5,
    "legend.frameon":True,"legend.framealpha":0.92,"legend.edgecolor":"#BDC3C7",
    "xtick.major.width":1.0,"ytick.major.width":1.0,
    "figure.dpi":80,"savefig.dpi":200,"savefig.bbox":"tight","savefig.facecolor":"white",
    "lines.linewidth":2.2,"patch.linewidth":1.2,"grid.linewidth":0.6,
})

# ============================================================
# 1. 工具
# ============================================================
def resolve_excel_file():
    for path in EXCEL_CANDIDATES:
        if not path:
            continue
        path_str = str(path).strip()
        if not path_str or path_str == ".":
            continue
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError("Could not find data_with_alkane.xlsx. Please set QSAR_EXCEL_FILE.")

def resolve_optional_file(candidates, label):
    for path in candidates:
        path = Path(path)
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {label}. Checked: {candidates}")

def label(ax,s,x=-0.12,y=1.04):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=16,fontweight="bold",va="top",ha="left")

def ev(yt,yp):
    return {"R2":r2_score(yt,yp),"RMSE":np.sqrt(mean_squared_error(yt,yp)),"MAE":mean_absolute_error(yt,yp)}

def gmetrics(yt,yp,mask):
    out={}
    for n,m in [("Alkane",mask),("Non-alkane",~mask)]:
        if m.sum()<2: out[n]={"R2":np.nan,"RMSE":np.nan,"MAE":np.nan,"n":int(m.sum())}
        else: out[n]={"R2":r2_score(yt[m],yp[m]),"RMSE":np.sqrt(mean_squared_error(yt[m],yp[m])),"MAE":mean_absolute_error(yt[m],yp[m]),"n":int(m.sum())}
    return out

def williams_ad(Xtr,ytr,Xte,yte,model):
    xtr=np.asarray(Xtr,float); xte=np.asarray(Xte,float)
    xtra=np.column_stack([np.ones(xtr.shape[0]),xtr]); xtea=np.column_stack([np.ones(xte.shape[0]),xte])
    Xinv=np.linalg.pinv(xtra.T@xtra)
    ltr=np.einsum("ij,jk,ik->i",xtra,Xinv,xtra); lte=np.einsum("ij,jk,ik->i",xtea,Xinv,xtea)
    rtr=ytr-model.predict(Xtr)
    s=max(float(np.sqrt(np.sum(rtr**2)/max(len(ytr)-xtra.shape[1],1))),1e-12)
    srtr=rtr/np.sqrt(np.maximum(s**2*(1-ltr),1e-12))
    rte=yte-model.predict(Xte); srte=rte/np.sqrt(np.maximum(s**2*(1-lte),1e-12))
    hstar=3*xtra.shape[1]/xtra.shape[0]
    return ltr,srtr,lte,srte,hstar

def nn_dist(Xtr,Xte):
    xtr=np.asarray(Xtr,float); xte=np.asarray(Xte,float)
    d2t=np.sum((xtr[:,None]-xtr[None,:])**2,2); np.fill_diagonal(d2t,np.inf); dt=np.sqrt(np.min(d2t,1))
    d2e=np.sum((xte[:,None]-xtr[None,:])**2,2); de=np.sqrt(np.min(d2e,1))
    return dt,de

def cq(scores,alpha):
    s=np.sort(np.asarray(scores,float)); n=len(s)
    if n==0: return np.nan
    r=int(np.ceil((n+1)*(1-alpha)))-1; return float(s[min(max(r,0),n-1)])

# ============================================================
# 2. 数据
# ============================================================
def load_data(excel_file):
    dfs,sysf=[],{}
    for sn,t in SYSTEMS.items():
        df=pd.read_excel(excel_file,sheet_name=sn)
        dc=[c for c in df.columns if c not in ["SMILES","molecule_names","is_alkane","System","Oxidant_E0","Target"] and not c.startswith("logk") and pd.api.types.is_numeric_dtype(df[c])]
        df=df.dropna(subset=dc+[t]).copy()
        df["System"]=sn; df["Oxidant_E0"]=OXIDANT_E0[sn]; df["Target"]=df[t]
        df["is_alkane"]=df["is_alkane"].astype(str).str.lower().isin(["1","true","yes","y","alkane"])
        sysf[sn]=df.copy()
        dfs.append(df[["molecule_names","is_alkane","System"]+dc+["Oxidant_E0","Target"]])
    data=pd.concat(dfs,axis=0).reset_index(drop=True)
    feats=[c for c in data.columns if c not in ["Target","molecule_names","is_alkane","System"]]
    return data,feats,sysf

def split_scale(data,feats):
    x=data[feats].values; y=data["Target"].values
    nm=data["molecule_names"].values; ak=data["is_alkane"].values.astype(bool); ss=data["System"].values
    xtr,xtmp,ytr,ytmp,ntr,ntmp,atr,atmp,strr,stmp=train_test_split(x,y,nm,ak,ss,train_size=TRAIN_SIZE,random_state=RANDOM_STATE,stratify=ss)
    vr=VAL_SIZE/(VAL_SIZE+TEST_SIZE)
    xv,xte,yv,yte,nv,nte,av,ate,sv,ste=train_test_split(xtmp,ytmp,ntmp,atmp,stmp,train_size=vr,random_state=RANDOM_STATE,stratify=stmp)
    sc=StandardScaler()
    return {"X_train":sc.fit_transform(xtr),"X_val":sc.transform(xv),"X_test":sc.transform(xte),
            "y_train":ytr,"y_val":yv,"y_test":yte,"names_train":ntr,"names_val":nv,"names_test":nte,
            "alk_train":atr,"alk_val":av,"alk_test":ate,"sys_train":strr,"sys_val":sv,"sys_test":ste,"scaler":sc}

def get_models():
    return {"Ridge":Ridge(alpha=1.0),"SVR":SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.1),
            "RF":RandomForestRegressor(n_estimators=400,random_state=RANDOM_STATE),
            "GBDT":GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=3,random_state=RANDOM_STATE)}

def train_all(sd):
    models=get_models(); mv={}; pv={}; pb=None; tgm={}
    for mn,m in models.items():
        m.fit(sd["X_train"],sd["y_train"])
        yvp=m.predict(sd["X_val"]); ytep=m.predict(sd["X_test"]); ytrp=m.predict(sd["X_train"])
        mv[mn]=ev(sd["y_val"],yvp); pv[mn]=yvp; tgm[mn]=gmetrics(sd["y_test"],ytep,sd["alk_test"])
        if mn=="GBDT": pb={"train":ytrp,"val":yvp,"test":ytep}
    best=max(mv,key=lambda k:mv[k]["R2"])
    return models,best,models[best],mv,pv,tgm,pb

def fit_acp(sd,ns=5,beta=1.0,alphas=(0.10,0.05)):
    xd=np.vstack([sd["X_train"],sd["X_val"]]); yd=np.concatenate([sd["y_train"],sd["y_val"]])
    sdv=np.concatenate([sd["sys_train"],sd["sys_val"]]); xte,yte=sd["X_test"],sd["y_test"]
    cr={}; sr=[]; sp=StratifiedKFold(n_splits=ns,shuffle=True,random_state=RANDOM_STATE)
    for mn,bm in get_models().items():
        oof=np.zeros(len(yd)); fms=[]
        for fi,hi in sp.split(xd,sdv):
            m=clone(bm); m.fit(xd[fi],yd[fi]); oof[hi]=m.predict(xd[hi]); fms.append(m)
        dfp=np.column_stack([m.predict(xd) for m in fms]); tfp=np.column_stack([m.predict(xte) for m in fms])
        sdd=dfp.std(1,ddof=0); sdt=tfp.std(1,ddof=0); tpm=tfp.mean(1); asc=np.abs(yd-oof)/(1.0+beta*sdd)
        ar={}
        for a in alphas:
            lv=int(round((1-a)*100)); qh=cq(asc,a); rt=qh*(1.0+beta*sdt)
            lo=tpm-rt; hi=tpm+rt; cv=(yte>=lo)&(yte<=hi)
            ar[lv]={"alpha":a,"qhat":qh,"lower_test":lo,"upper_test":hi,
                     "coverage":float(np.mean(cv)),"mean_width":float(np.mean(hi-lo)),"covered":cv}
            sr.append({"Model":mn,"Interval":f"{lv}%","Empirical_Coverage_Test":float(np.mean(cv)),
                        "Mean_Width_Test":float(np.mean(hi-lo)),"qhat":qh})
        cr[mn]={"y_test":yte,"test_pred_mean":tpm,"sigma_test":sdt,"intervals":ar,
                "test_metrics":ev(yte,tpm),"sys_test":sd["sys_test"],"alk_test":sd["alk_test"],
                "names_test":sd["names_test"]}
    return cr,pd.DataFrame(sr)

# ============================================================
# 2b. Web inference & extended utility functions (from paper_aligned)
# ============================================================

def evaluate_model(y_true, y_pred, name=""):
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    print(f"{name} | R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}")
    return metrics

def safe_group_metrics(y_true, y_pred, group_flag):
    out = {}
    for group_name, mask in {"Alkane": group_flag, "Non-alkane": ~group_flag}.items():
        if np.sum(mask) < 2:
            out[group_name] = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": int(np.sum(mask))}
        else:
            out[group_name] = {
                "R2": r2_score(y_true[mask], y_pred[mask]),
                "RMSE": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
                "n": int(np.sum(mask)),
            }
    return out

def print_metric_block(title, metrics_dict):
    print(f"\n{title}")
    for model_name, metrics in metrics_dict.items():
        print(
            f"  {model_name:<6} | "
            f"R2={metrics['R2']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"MAE={metrics['MAE']:.3f}"
        )

def print_group_metric_block(title, group_metrics_dict):
    print(f"\n{title}")
    for model_name, groups in group_metrics_dict.items():
        print(f"  {model_name}:")
        for group_name in ["Alkane", "Non-alkane"]:
            metrics = groups[group_name]
            if np.isnan(metrics["R2"]):
                print(f"    {group_name:<12} | n={metrics['n']}, insufficient samples")
            else:
                print(
                    f"    {group_name:<12} | "
                    f"n={metrics['n']}, "
                    f"R2={metrics['R2']:.3f}, "
                    f"RMSE={metrics['RMSE']:.3f}, "
                    f"MAE={metrics['MAE']:.3f}"
                )

def williams_ad_arrays(X_train, y_train, X_test, y_test, model):
    xtr = np.asarray(X_train, dtype=float)
    xte = np.asarray(X_test, dtype=float)
    xtr_aug = np.column_stack([np.ones(xtr.shape[0]), xtr])
    xte_aug = np.column_stack([np.ones(xte.shape[0]), xte])
    xtx_inv = np.linalg.pinv(xtr_aug.T @ xtr_aug)
    lev_tr = np.einsum("ij,jk,ik->i", xtr_aug, xtx_inv, xtr_aug)
    lev_te = np.einsum("ij,jk,ik->i", xte_aug, xtx_inv, xte_aug)
    resid_train = y_train - model.predict(X_train)
    sigma = np.sqrt(np.sum(resid_train ** 2) / max(len(y_train) - xtr_aug.shape[1], 1))
    sigma = max(float(sigma), 1e-12)
    sres_tr = resid_train / np.sqrt(np.maximum(sigma ** 2 * (1 - lev_tr), 1e-12))
    resid_test = y_test - model.predict(X_test)
    sres_te = resid_test / np.sqrt(np.maximum(sigma ** 2 * (1 - lev_te), 1e-12))
    h_star = 3 * xtr_aug.shape[1] / xtr_aug.shape[0]
    return lev_tr, sres_tr, lev_te, sres_te, h_star

def nearest_training_distances(X_train, X_test):
    xtr = np.asarray(X_train, dtype=float)
    xte = np.asarray(X_test, dtype=float)
    train_diff = xtr[:, None, :] - xtr[None, :, :]
    train_d2 = np.sum(train_diff ** 2, axis=2)
    np.fill_diagonal(train_d2, np.inf)
    d_train = np.sqrt(np.min(train_d2, axis=1))
    test_diff = xte[:, None, :] - xtr[None, :, :]
    test_d2 = np.sum(test_diff ** 2, axis=2)
    d_test = np.sqrt(np.min(test_d2, axis=1))
    return d_train, d_test

def make_shap_explainer(model, x_background):
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        return shap.TreeExplainer(model)
    background = shap.sample(x_background, min(100, len(x_background)), random_state=RANDOM_STATE)
    return shap.Explainer(model.predict, background)

def get_shap_values(explainer, x_data):
    out = explainer(x_data)
    return np.asarray(out.values) if hasattr(out, "values") else np.asarray(out)

def load_combined_dataset(excel_file):
    dfs = []
    system_frames = {}
    for sys_name, target in SYSTEMS.items():
        df = pd.read_excel(excel_file, sheet_name=sys_name)
        descriptor_cols = [
            c for c in df.columns
            if c not in ["SMILES", "molecule_names", "is_alkane", "System", "Oxidant_E0", "Target"]
            and not c.startswith("logk")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = df.dropna(subset=descriptor_cols + [target]).copy()
        df["System"] = sys_name
        df["Oxidant_E0"] = OXIDANT_E0[sys_name]
        df["Target"] = df[target]
        df["is_alkane"] = df["is_alkane"].astype(str).str.lower().isin(["1", "true", "yes", "y", "alkane"])
        system_frames[sys_name] = df.copy()
        dfs.append(df[["molecule_names", "is_alkane", "System"] + descriptor_cols + ["Oxidant_E0", "Target"]])
    data = pd.concat(dfs, axis=0).reset_index(drop=True)
    feature_cols = [c for c in data.columns if c not in ["Target", "molecule_names", "is_alkane", "System"]]
    return data, feature_cols, system_frames

def split_and_scale(data, feature_cols):
    x = data[feature_cols].values
    y = data["Target"].values
    names = data["molecule_names"].values
    alk = data["is_alkane"].values.astype(bool)
    systems = data["System"].values
    x_train, x_tmp, y_train, y_tmp, names_train, names_tmp, alk_train, alk_tmp, sys_train, sys_tmp = train_test_split(
        x, y, names, alk, systems, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=systems
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    x_val, x_test, y_val, y_test, names_val, names_test, alk_val, alk_test, sys_val, sys_test = train_test_split(
        x_tmp, y_tmp, names_tmp, alk_tmp, sys_tmp, train_size=val_ratio, random_state=RANDOM_STATE, stratify=sys_tmp
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    return {
        "X_train": x_train_s, "X_val": x_val_s, "X_test": x_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "names_train": names_train, "names_val": names_val, "names_test": names_test,
        "alk_train": alk_train, "alk_val": alk_val, "alk_test": alk_test,
        "sys_train": sys_train, "sys_val": sys_val, "sys_test": sys_test,
        "scaler": scaler,
    }

def get_model_dict():
    return {
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
        "RF": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "GBDT": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return np.nan
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(scores[rank])

def fit_adaptive_conformal_models(split_data, n_splits=5, beta=1.0, alphas=(0.10, 0.05)):
    x_dev = np.vstack([split_data["X_train"], split_data["X_val"]])
    y_dev = np.concatenate([split_data["y_train"], split_data["y_val"]])
    sys_dev = np.concatenate([split_data["sys_train"], split_data["sys_val"]])
    names_dev = np.concatenate([split_data["names_train"], split_data["names_val"]])
    alk_dev = np.concatenate([split_data["alk_train"], split_data["alk_val"]])
    x_test = split_data["X_test"]
    y_test = split_data["y_test"]
    conformal_results = {}
    summary_rows = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for model_name, base_model in get_model_dict().items():
        oof_pred = np.zeros(len(y_dev), dtype=float)
        fold_models = []
        for fold_id, (fit_idx, hold_idx) in enumerate(splitter.split(x_dev, sys_dev), start=1):
            model = clone(base_model)
            model.fit(x_dev[fit_idx], y_dev[fit_idx])
            oof_pred[hold_idx] = model.predict(x_dev[hold_idx])
            fold_models.append(model)
            print(f"Conformal-{model_name} | finished fold {fold_id}/{n_splits}")
        dev_fold_preds = np.column_stack([model.predict(x_dev) for model in fold_models])
        test_fold_preds = np.column_stack([model.predict(x_test) for model in fold_models])
        sigma_dev = dev_fold_preds.std(axis=1, ddof=0)
        sigma_test = test_fold_preds.std(axis=1, ddof=0)
        test_pred_mean = test_fold_preds.mean(axis=1)
        adaptive_scores = np.abs(y_dev - oof_pred) / (1.0 + beta * sigma_dev)
        alpha_results = {}
        for alpha in alphas:
            level = int(round((1 - alpha) * 100))
            qhat = conformal_quantile(adaptive_scores, alpha)
            radius_test = qhat * (1.0 + beta * sigma_test)
            lower_test = test_pred_mean - radius_test
            upper_test = test_pred_mean + radius_test
            covered = (y_test >= lower_test) & (y_test <= upper_test)
            coverage = float(np.mean(covered))
            mean_width = float(np.mean(upper_test - lower_test))
            alpha_results[level] = {
                "alpha": alpha, "qhat": qhat,
                "lower_test": lower_test, "upper_test": upper_test,
                "radius_test": radius_test,
                "coverage": coverage, "mean_width": mean_width, "covered": covered,
            }
            summary_rows.append({
                "Model": model_name, "Interval": f"{level}%",
                "Nominal_Coverage": 1 - alpha,
                "Empirical_Coverage_Test": coverage,
                "Mean_Width_Test": mean_width,
                "Median_Width_Test": float(np.median(upper_test - lower_test)),
                "qhat": qhat, "beta": beta,
                "R2_Test": r2_score(y_test, test_pred_mean),
                "RMSE_Test": np.sqrt(mean_squared_error(y_test, test_pred_mean)),
                "MAE_Test": mean_absolute_error(y_test, test_pred_mean),
            })
        conformal_results[model_name] = {
            "fold_models": fold_models,
            "oof_pred_dev": oof_pred,
            "sigma_dev": sigma_dev,
            "adaptive_scores": adaptive_scores,
            "dev_pred_mean": dev_fold_preds.mean(axis=1),
            "test_pred_mean": test_pred_mean,
            "sigma_test": sigma_test,
            "y_dev": y_dev, "y_test": y_test,
            "names_dev": names_dev, "names_test": split_data["names_test"],
            "alk_dev": alk_dev, "alk_test": split_data["alk_test"],
            "sys_dev": sys_dev, "sys_test": split_data["sys_test"],
            "intervals": alpha_results,
            "test_metrics": evaluate_model(y_test, test_pred_mean, f"Conformal-Test-{model_name}"),
        }
    summary_df = pd.DataFrame(summary_rows)
    return conformal_results, summary_df

def infer_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    if isinstance(model, SVR):
        return "SVR"
    if isinstance(model, RandomForestRegressor):
        return "RF"
    if isinstance(model, GradientBoostingRegressor):
        return "GBDT"
    raise ValueError(f"Unsupported model type for web inference: {type(model)!r}")

def fit_single_model_adaptive_conformal(split_data, model_name, n_splits=5, beta=1.0, alphas=(0.10, 0.05)):
    if model_name not in get_model_dict():
        raise KeyError(f"Unknown model name: {model_name}")
    x_dev = np.vstack([split_data["X_train"], split_data["X_val"]])
    y_dev = np.concatenate([split_data["y_train"], split_data["y_val"]])
    sys_dev = np.concatenate([split_data["sys_train"], split_data["sys_val"]])
    base_model = get_model_dict()[model_name]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y_dev), dtype=float)
    fold_models = []
    for fit_idx, hold_idx in splitter.split(x_dev, sys_dev):
        fold_model = clone(base_model)
        fold_model.fit(x_dev[fit_idx], y_dev[fit_idx])
        oof_pred[hold_idx] = fold_model.predict(x_dev[hold_idx])
        fold_models.append(fold_model)
    dev_fold_preds = np.column_stack([model.predict(x_dev) for model in fold_models])
    sigma_dev = dev_fold_preds.std(axis=1, ddof=0)
    adaptive_scores = np.abs(y_dev - oof_pred) / (1.0 + beta * sigma_dev)
    intervals = {}
    for alpha in alphas:
        level = int(round((1 - alpha) * 100))
        intervals[level] = {
            "alpha": alpha,
            "qhat": conformal_quantile(adaptive_scores, alpha),
        }
    return {
        "model_name": model_name,
        "beta": beta,
        "n_splits": n_splits,
        "intervals": intervals,
        "fold_models": fold_models,
        "feature_order": split_data["features"],
    }

def predict_with_conformal_bundle(conformal_bundle, X_scaled):
    x_scaled = np.asarray(X_scaled, dtype=float)
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(1, -1)
    fold_models = conformal_bundle["fold_models"]
    fold_preds = np.column_stack([model.predict(x_scaled) for model in fold_models])
    pred_mean = fold_preds.mean(axis=1)
    sigma = fold_preds.std(axis=1, ddof=0)
    beta = float(conformal_bundle.get("beta", 1.0))
    intervals = {}
    for level, interval_cfg in sorted(conformal_bundle["intervals"].items()):
        qhat = float(interval_cfg["qhat"])
        radius = qhat * (1.0 + beta * sigma)
        intervals[int(level)] = {
            "lower": pred_mean - radius,
            "upper": pred_mean + radius,
            "width": 2.0 * radius,
            "qhat": qhat,
        }
    return {
        "pred_mean": pred_mean,
        "sigma": sigma,
        "intervals": intervals,
    }

def save_web_inference_artifacts(model, scaler, features, conformal_bundle, model_dir=None):
    target_dir = Path(model_dir) if model_dir is not None else WEB_MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_dir / "unified_qsar_model.pkl")
    joblib.dump(scaler, target_dir / "scaler.pkl")
    joblib.dump(conformal_bundle, target_dir / "conformal_predictor.pkl")
    with open(target_dir / "feature_order.txt", "w", encoding="utf-8") as f:
        for feat in features:
            f.write(feat + "\n")
    metadata = {
        "model_name": infer_model_name(model),
        "feature_count": len(features),
        "beta": float(conformal_bundle.get("beta", 1.0)),
        "interval_levels": sorted(int(level) for level in conformal_bundle["intervals"].keys()),
    }
    with open(target_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return target_dir

def build_and_save_web_inference_assets(model_dir=None, n_splits=5, beta=1.0, alphas=(0.10, 0.05)):
    ensure_dirs()
    excel_file = resolve_excel_file()
    data, features, _ = load_combined_dataset(excel_file)
    split_data = split_and_scale(data, features)
    split_data["features"] = features
    _, best_name, best_model, _, _, _, _ = train_models(split_data)
    conformal_bundle = fit_single_model_adaptive_conformal(
        split_data, model_name=best_name, n_splits=n_splits, beta=beta, alphas=alphas,
    )
    save_web_inference_artifacts(best_model, split_data["scaler"], features, conformal_bundle, model_dir=model_dir)
    return {
        "model_name": best_name,
        "features": features,
        "model_dir": str(Path(model_dir) if model_dir is not None else WEB_MODEL_DIR),
    }

def load_web_inference_assets(model_dir=None):
    source_dir = Path(model_dir) if model_dir is not None else WEB_MODEL_DIR
    model = joblib.load(source_dir / "unified_qsar_model.pkl")
    scaler = joblib.load(source_dir / "scaler.pkl")
    conformal_bundle = joblib.load(source_dir / "conformal_predictor.pkl")
    with open(source_dir / "feature_order.txt", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]
    metadata_path = source_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_name": infer_model_name(model),
            "feature_count": len(features),
            "interval_levels": sorted(int(level) for level in conformal_bundle["intervals"].keys()),
        }
    return {
        "model": model, "scaler": scaler, "features": features,
        "conformal_bundle": conformal_bundle, "metadata": metadata,
        "model_dir": source_dir,
    }

def ensure_web_inference_assets(model_dir=None):
    source_dir = Path(model_dir) if model_dir is not None else WEB_MODEL_DIR
    required_files = [
        source_dir / "unified_qsar_model.pkl",
        source_dir / "scaler.pkl",
        source_dir / "feature_order.txt",
        source_dir / "conformal_predictor.pkl",
        source_dir / "model_metadata.json",
    ]
    if all(path.exists() for path in required_files):
        return load_web_inference_assets(source_dir)
    build_and_save_web_inference_assets(model_dir=source_dir)
    return load_web_inference_assets(source_dir)

def train_models(split_data):
    models = get_model_dict()
    metrics_train = {}
    metrics_val = {}
    metrics_test = {}
    preds_train = {}
    preds_val = {}
    preds_test = {}
    test_group_metrics = {}
    for model_name, model in models.items():
        model.fit(split_data["X_train"], split_data["y_train"])
        y_train_pred = model.predict(split_data["X_train"])
        y_val_pred = model.predict(split_data["X_val"])
        y_test_pred = model.predict(split_data["X_test"])
        metrics_train[model_name] = evaluate_model(split_data["y_train"], y_train_pred, f"Train-{model_name}")
        metrics_val[model_name] = evaluate_model(split_data["y_val"], y_val_pred, f"Val-{model_name}")
        metrics_test[model_name] = evaluate_model(split_data["y_test"], y_test_pred, f"Test-{model_name}")
        preds_train[model_name] = y_train_pred
        preds_val[model_name] = y_val_pred
        preds_test[model_name] = y_test_pred
        test_group_metrics[model_name] = safe_group_metrics(split_data["y_test"], y_test_pred, split_data["alk_test"])
    print_metric_block("Training Set Metrics", metrics_train)
    print_metric_block("Validation Set Metrics", metrics_val)
    print_metric_block("Test Set Metrics", metrics_test)
    print_group_metric_block("Test Set Group Metrics", test_group_metrics)
    best_name = max(metrics_val, key=lambda k: metrics_val[k]["R2"])
    best_model = models[best_name]
    preds_best = {
        "train": preds_train[best_name],
        "val": preds_val[best_name],
        "test": preds_test[best_name],
    }
    print(f"\nBest model: {best_name}")
    evaluate_model(split_data["y_train"], preds_best["train"], f"{best_name}-Train")
    evaluate_model(split_data["y_val"], preds_best["val"], f"{best_name}-Val")
    evaluate_model(split_data["y_test"], preds_best["test"], f"{best_name}-Test")
    return models, best_name, best_model, metrics_val, preds_val, test_group_metrics, preds_best

def save_conformal_outputs(conformal_results, summary_df):
    summary_path = CONFORMAL_DIR / "Adaptive_Conformal_Summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    for model_name, result in conformal_results.items():
        out_df = pd.DataFrame({
            "Molecule": result["names_test"],
            "System": result["sys_test"],
            "Compound_Type": np.where(result["alk_test"], "Alkane", "Non-alkane"),
            "Observed": result["y_test"],
            "Pred_Mean": result["test_pred_mean"],
            "Sigma": result["sigma_test"],
        })
        for level, interval_data in sorted(result["intervals"].items()):
            out_df[f"Lower_{level}"] = interval_data["lower_test"]
            out_df[f"Upper_{level}"] = interval_data["upper_test"]
            out_df[f"Width_{level}"] = interval_data["upper_test"] - interval_data["lower_test"]
            out_df[f"Covered_{level}"] = interval_data["covered"]
        out_df.to_csv(CONFORMAL_DIR / f"Adaptive_Conformal_Test_{model_name}.csv", index=False, encoding="utf-8-sig")
    return summary_path

# ============================================================
# 3. 6张图
# ============================================================

# ---------- Fig.1 ----------
def plot_fig1(data,split_data,system_frames):
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1,2,figure=fig,wspace=0.30,left=0.07,right=0.96,top=0.92,bottom=0.12)

    # (a) 三体系KDE
    ax = fig.add_subplot(gs[0,0])
    for sn in SYSTEMS:
        vals = system_frames[sn][SYSTEMS[sn]].values
        ax.hist(vals,bins=25,density=True,alpha=0.18,color=S_COLORS[sn],edgecolor="none")
        xk=np.linspace(vals.min()-0.5,vals.max()+0.5,300); ky=gaussian_kde(vals)(xk)
        ax.plot(xk,ky,color=S_COLORS[sn],lw=2.8,label=f"{sn} ($n$={len(vals)})")
        ax.fill_between(xk,ky,alpha=0.06,color=S_COLORS[sn])
    ax.set_xlabel("log$k$ (M$^{-1}$s$^{-1}$)",fontsize=14,fontweight="bold")
    ax.set_ylabel("Probability Density",fontsize=14,fontweight="bold")
    ax.set_title("log$k$ Distribution by Oxidant System",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,loc="upper left",framealpha=0.9); ax.tick_params(labelsize=12)
    label(ax,"(a)")

    # (b) 箱线+散点 — 图例右下角
    ax = fig.add_subplot(gs[0,1])
    box_alk,box_nalk=[],[]
    for sn,t in SYSTEMS.items():
        df=system_frames[sn]; box_alk.append(df.loc[df["is_alkane"],t].values); box_nalk.append(df.loc[~df["is_alkane"],t].values)
    pos_a,pos_n=[1,4,7],[2,5,8]
    ax.boxplot(box_alk,positions=pos_a,widths=0.72,patch_artist=True,
               boxprops=dict(facecolor=C_ORANGE_L,edgecolor=C_ORANGE,lw=1.5),
               medianprops=dict(color=C_RED,lw=2.8),whiskerprops=dict(color=C_ORANGE,lw=1.2),
               capprops=dict(color=C_ORANGE,lw=1.2),flierprops=dict(marker='o',mfc=C_ORANGE,ms=5,alpha=0.6))
    ax.boxplot(box_nalk,positions=pos_n,widths=0.72,patch_artist=True,
               boxprops=dict(facecolor=C_LIGHT,edgecolor=C_ACCENT,lw=1.5),
               medianprops=dict(color=C_RED,lw=2.8),whiskerprops=dict(color=C_ACCENT,lw=1.2),
               capprops=dict(color=C_ACCENT,lw=1.2),flierprops=dict(marker='o',mfc=C_ACCENT,ms=5,alpha=0.6))
    rng=np.random.default_rng(42)
    for pl,bd,col in [(pos_a,box_alk,C_ORANGE),(pos_n,box_nalk,C_ACCENT)]:
        for p,v in zip(pl,bd):
            if len(v)>0: j=rng.uniform(-0.16,0.16,len(v)); ax.scatter(np.full(len(v),p)+j,v,color=col,s=10,alpha=0.50,edgecolors='none')
    ax.set_xticks([1.5,4.5,7.5]); ax.set_xticklabels(list(SYSTEMS.keys()),fontsize=13,fontweight="bold")
    ax.set_ylabel("log$k$ (M$^{-1}$s$^{-1}$)",fontsize=14,fontweight="bold")
    ax.set_title("log$k$ by Oxidant System & Compound Type",fontsize=15,fontweight="bold")
    ax.legend(handles=[mpatches.Patch(facecolor=C_ORANGE_L,edgecolor=C_ORANGE,label="Alkane"),
                        mpatches.Patch(facecolor=C_LIGHT,edgecolor=C_ACCENT,label="Non-alkane")],
              fontsize=11,loc="lower right",framealpha=0.9)
    ax.tick_params(labelsize=12); label(ax,"(b)")

    fig.savefig(RESULT_DIR/"Fig1_Data_Overview.png"); plt.close(fig)
    print("[OK] Fig1_Data_Overview.png")

# ---------- Fig.2 ----------
def plot_fig2(metrics_val,split_data,preds_val,preds_best,best_name):
    fig = plt.figure(figsize=(16,12))
    gs = gridspec.GridSpec(2,2,figure=fig,hspace=0.32,wspace=0.28,left=0.07,right=0.96,top=0.94,bottom=0.06)
    mn_list=["Ridge","SVR","RF","GBDT"]
    r2s=[metrics_val[m]["R2"] for m in mn_list]; rmses=[metrics_val[m]["RMSE"] for m in mn_list]; maes=[metrics_val[m]["MAE"] for m in mn_list]

    # (a) 柱状 — 图例右上, 单列
    ax = fig.add_subplot(gs[0,0])
    x=np.arange(len(mn_list)); w=0.25
    b1=ax.bar(x-w,r2s,w,label="$R^2$ (↑)",color="#4a7ba7",alpha=0.88,edgecolor="white",lw=0.5)
    b2=ax.bar(x,rmses,w,label="RMSE (↓)",color="#ed9e5c",alpha=0.88,edgecolor="white",lw=0.5)
    b3=ax.bar(x+w,maes,w,label="MAE (↓)",color="#48a36a",alpha=0.88,edgecolor="white",lw=0.5)
    for bars in [b1,b2,b3]:
        for bar in bars:
            h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+0.015,f"{h:.3f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(mn_list,fontsize=13,fontweight="bold")
    ax.set_ylabel("Score",fontsize=14,fontweight="bold")
    ax.set_title("Model Comparison (Validation Set)",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,ncol=1,loc="upper right",framealpha=0.95)
    ax.set_ylim(0,max(max(r2s),max(rmses),max(maes))*1.22); ax.tick_params(labelsize=12)
    label(ax,"(a)")

    # (b) GBDT parity — 统一尺寸
    ax = fig.add_subplot(gs[0,1])
    all_obs=np.concatenate([split_data["y_train"],split_data["y_val"],split_data["y_test"]])
    all_pred=np.concatenate([preds_best["train"],preds_best["val"],preds_best["test"]])
    lo=min(all_obs.min(),all_pred.min())-0.4; hi=max(all_obs.max(),all_pred.max())+0.4
    for lbl,yt,yp,col,mk,sz,al in [("Train",split_data["y_train"],preds_best["train"],C_PRIMARY,"o",22,0.38),
        ("Val",split_data["y_val"],preds_best["val"],C_ORANGE,"s",30,0.58),
        ("Test",split_data["y_test"],preds_best["test"],C_GREEN,"D",38,0.82)]:
        r2v=r2_score(yt,yp); rmsev=np.sqrt(mean_squared_error(yt,yp))
        ax.scatter(yt,yp,color=col,marker=mk,s=sz,alpha=al,edgecolors="white",lw=0.3,
                   label=f"{lbl} ($R^2$={r2v:.3f}, RMSE={rmsev:.3f})")
    ax.plot([lo,hi],[lo,hi],color=C_RED,lw=2.0,ls="--",alpha=0.8,zorder=0)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel("Observed log$k$",fontsize=14,fontweight="bold")
    ax.set_ylabel("Predicted log$k$",fontsize=14,fontweight="bold")
    ax.set_title(f"Parity Plot — {best_name} (All Splits)",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,loc="lower right",framealpha=0.9)
    ax.tick_params(labelsize=12)
    label(ax,"(b)")

    # (c) CDF
    ax = fig.add_subplot(gs[1,0])
    train_err=np.abs(split_data["y_train"]-preds_best["train"])
    val_err=np.abs(split_data["y_val"]-preds_best["val"])
    test_err=np.abs(split_data["y_test"]-preds_best["test"])
    for errs,col,lbl in [(train_err,C_PRIMARY,"Train"),(val_err,C_ORANGE,"Val"),(test_err,C_GREEN,"Test")]:
        xs=np.sort(errs); yc=np.arange(1,len(xs)+1)/len(xs)
        ax.plot(xs,yc,color=col,lw=2.8,label=lbl); ax.fill_between(xs,0,yc,color=col,alpha=0.07)
        p90=np.percentile(errs,90); ax.axvline(p90,color=col,lw=1.4,ls=":",alpha=0.6)
        ax.text(p90+0.02,0.12+0.06*(["Train","Val","Test"].index(lbl)),f"P90={p90:.3f}",color=col,fontsize=10,rotation=90,va="bottom",fontweight="bold")
    ax.axhline(0.90,color=C_GRAY,lw=1.2,ls="--",alpha=0.5)
    ax.set_xlabel("|Residual| (Absolute Error)",fontsize=14,fontweight="bold")
    ax.set_ylabel("Cumulative Fraction",fontsize=14,fontweight="bold")
    ax.set_title("Cumulative Error Distribution (CDF)",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,loc="lower right",framealpha=0.9)
    ax.set_xlim(left=0); ax.set_ylim(0,1.02); ax.grid(True,alpha=0.15,lw=0.5); ax.tick_params(labelsize=12)
    label(ax,"(c)")
    print(f"  CDF P90: Train={np.percentile(train_err,90):.3f}, Val={np.percentile(val_err,90):.3f}, Test={np.percentile(test_err,90):.3f}")

    # (d) 残差
    ax = fig.add_subplot(gs[1,1])
    train_resid=split_data["y_train"]-preds_best["train"]; test_resid=split_data["y_test"]-preds_best["test"]
    ax.scatter(preds_best["train"],train_resid,color=C_PRIMARY,s=24,alpha=0.38,edgecolors="none")
    ax.scatter(preds_best["test"],test_resid,color=C_ORANGE,s=40,alpha=0.80,edgecolors="white",lw=0.3,marker="^")
    ax.axhline(0,color=C_RED,lw=2.0,ls="--",alpha=0.8); ax.axhline(+2,color=C_GRAY,lw=1.0,ls=":",alpha=0.5); ax.axhline(-2,color=C_GRAY,lw=1.0,ls=":",alpha=0.5)
    std_tr=np.std(train_resid); std_te=np.std(test_resid)
    ax.axhline(+2*std_tr,color=C_PRIMARY,lw=1.0,ls="-.",alpha=0.4); ax.axhline(-2*std_tr,color=C_PRIMARY,lw=1.0,ls="-.",alpha=0.4)
    ax.set_xlabel("Predicted log$k$",fontsize=14,fontweight="bold")
    ax.set_ylabel("Residual (Observed - Predicted)",fontsize=14,fontweight="bold")
    ax.set_title("Residual Analysis",fontsize=15,fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.text(0.97,0.06,f"Train (blue): σ = {std_tr:.3f}\nTest (orange): σ = {std_te:.3f}",
            transform=ax.transAxes,fontsize=11,va="bottom",ha="right",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec=C_GRAY,alpha=0.92))
    label(ax,"(d)")

    fig.savefig(RESULT_DIR/"Fig2_Model_GBDT_Combined.png"); plt.close(fig)
    print("[OK] Fig2_Model_GBDT_Combined.png")

# ---------- Fig.3 (共形预测, 颜色与Fig.1一致) ----------
def plot_fig3(conformal_results,summary_df):
    model_names=list(conformal_results.keys())
    fig = plt.figure(figsize=(17,12))
    gs = gridspec.GridSpec(3,2,figure=fig,hspace=0.35,wspace=0.24,left=0.06,right=0.97,top=0.95,bottom=0.06)

    for idx,model_name in enumerate(model_names):
        row,col=divmod(idx,2); ax=fig.add_subplot(gs[row,col])
        result=conformal_results[model_name]; order=np.argsort(result["y_test"]); x_axis=np.arange(len(order))
        y_obs=result["y_test"][order]; y_hat=result["test_pred_mean"][order]
        d90=result["intervals"][90]; d95=result["intervals"][95]

        ax.fill_between(x_axis,d95["lower_test"][order],d95["upper_test"][order],
                        color=C_LIGHT,alpha=0.40,label=f"95% PI (Cov={d95['coverage']:.2f})")
        ax.fill_between(x_axis,d90["lower_test"][order],d90["upper_test"][order],
                        color=C_ACCENT,alpha=0.22,label=f"90% PI (Cov={d90['coverage']:.2f})")
        ax.plot(x_axis,y_hat,color=M_COLORS[model_name],lw=2.5,label="Predicted mean")
        ax.scatter(x_axis,y_obs,s=20,color=C_RED,alpha=0.90,edgecolors="white",lw=0.3,zorder=5,label="Observed")
        ax.set_title(f"{model_name} - Adaptive Conformal Intervals",fontsize=15,fontweight="bold")
        ax.set_xlabel("Test Samples (sorted by observed log$k$)",fontsize=13,fontweight="bold")
        ax.set_ylabel("log$k$",fontsize=14,fontweight="bold")
        ax.legend(fontsize=9,loc="lower right",ncol=2,framealpha=0.9); ax.tick_params(labelsize=11)
        label(ax,f"({chr(97+idx)})")

    # (e) 覆盖率 — 使用统一配色
    ax = fig.add_subplot(gs[2,0]); x=np.arange(len(model_names)); w=0.20
    for j,il in enumerate(["90%","95%"]):
        vals=[summary_df[(summary_df["Model"]==m)&(summary_df["Interval"]==il)]["Empirical_Coverage_Test"].iloc[0] for m in model_names]
        bars=ax.bar(x+(j-0.5)*w,vals,w,label=f"{il} (target={float(il.strip('%'))/100:.2f})",alpha=0.88,
                    color=["#ed9e5c","#4a7ba7"][j],edgecolor="white",lw=0.5)
        ax.axhline(float(il.strip("%"))/100.0,color=["#ed9e5c","#4a7ba7"][j],lw=1.5,ls="--",alpha=0.6)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.008,f"{v:.3f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_names,fontsize=13,fontweight="bold"); ax.set_ylim(0,1.08)
    ax.set_ylabel("Empirical Coverage",fontsize=14,fontweight="bold")
    ax.set_title("Coverage Comparison",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,loc="lower right",framealpha=0.9); ax.tick_params(labelsize=12); label(ax,"(e)")

    # (f) 宽度 — 使用统一配色
    ax = fig.add_subplot(gs[2,1])
    for j,il in enumerate(["90%","95%"]):
        vals=[summary_df[(summary_df["Model"]==m)&(summary_df["Interval"]==il)]["Mean_Width_Test"].iloc[0] for m in model_names]
        bars=ax.bar(x+(j-0.5)*w,vals,w,label=f"{il} Mean Width",alpha=0.88,
                    color=["#48a36a","#ed9e5c"][j],edgecolor="white",lw=0.5)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f"{v:.3f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_names,fontsize=13,fontweight="bold")
    ax.set_ylabel("Mean Interval Width",fontsize=14,fontweight="bold")
    ax.set_title("Sharpness Comparison (Lower = Better)",fontsize=15,fontweight="bold")
    ax.legend(fontsize=11,loc="upper right",framealpha=0.9); ax.tick_params(labelsize=12); label(ax,"(f)")

    fig.savefig(RESULT_DIR/"Fig3_Conformal_Prediction.png"); plt.close(fig)
    print("[OK] Fig3_Conformal_Prediction.png")

# ---------- Fig.4 (SHAP, 颜色优化) ----------
def plot_fig4(best_model,split_data,df_imp,df_compare):
    explainer = shap.TreeExplainer(best_model)
    sv_train=explainer(split_data["X_train"]).values; sv_test=explainer(split_data["X_test"]).values
    features=split_data["features"]

    fig = plt.figure(figsize=(16,11))
    gs = gridspec.GridSpec(2,2,figure=fig,hspace=0.34,wspace=0.30,left=0.07,right=0.97,top=0.94,bottom=0.07)

    # (a) 重要性 — 用不同颜色区分
    ax = fig.add_subplot(gs[0,0])
    top_n=min(12,len(features)); top_df=df_imp.head(top_n).iloc[::-1]
    colors_a=["#5ab07b","#5ab07b","#5ab07b","#8b6aad","#8b6aad","#8b6aad","#4a7ba7","#4a7ba7","#4a7ba7","#ed9e5c","#ed9e5c","#ed9e5c"]
    bars=ax.barh(range(top_n),top_df["Importance"].values,color=colors_a[:top_n],edgecolor="white",lw=0.5,height=0.70)
    for i,(bar,v) in enumerate(zip(bars,top_df["Importance"].values)):
        ax.text(v+top_df["Importance"].max()*0.02,bar.get_y()+bar.get_height()/2,f"{v:.3f}",va="center",fontsize=11,fontweight="bold")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_df["Feature"].values,fontsize=11,fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|",fontsize=14,fontweight="bold")
    ax.set_title("Feature Importance (Top-12)",fontsize=15,fontweight="bold"); ax.tick_params(labelsize=12)
    label(ax,"(a)")

    # (b) 蜂群
    ax = fig.add_subplot(gs[0,1])
    top_shap=min(10,len(features)); top_idx_b=df_imp.head(top_shap).index.tolist(); feat_order=list(reversed(top_idx_b))
    rng=np.random.default_rng(42)
    for yi,fi in enumerate(feat_order):
        sv=sv_train[:,fi]; fval=split_data["X_train"][:,fi]; fv_norm=(fval-fval.min())/(fval.max()-fval.min()+1e-9)
        jit=rng.uniform(-0.28,0.28,size=len(sv))
        sc=ax.scatter(sv,yi+jit,c=fv_norm,cmap=CMAP_BWOR,s=9,alpha=0.58,vmin=0,vmax=1,edgecolors="none")
    ax.axvline(0,color=C_GRAY,lw=1.4,ls="--",alpha=0.7)
    ax.set_yticks(range(top_shap)); ax.set_yticklabels([features[i] for i in feat_order],fontsize=11,fontweight="bold")
    ax.set_xlabel("SHAP Value",fontsize=14,fontweight="bold")
    ax.set_title("SHAP Beeswarm (Training Set)",fontsize=15,fontweight="bold")
    cbar=plt.colorbar(sc,ax=ax,fraction=0.04,pad=0.04); cbar.set_label("Feature Value (Low -> High)",fontsize=10,fontweight="bold"); cbar.ax.tick_params(labelsize=9)
    ax.tick_params(labelsize=12); label(ax,"(b)")

    # (c) 烷烃 vs 非烷烃 — 按总重要性从长到短排列
    ax = fig.add_subplot(gs[1,0])
    top_comp=min(10,len(df_compare)); comp_df=df_compare.copy()
    comp_df["total"]=comp_df["Alkane"]+comp_df["Non_alkane"]
    comp_df=comp_df.sort_values("total",ascending=True).head(top_comp)  # ascending for barh(bottom=shortest)
    cx=np.arange(top_comp); cw=0.36
    ax.barh(cx-cw/2,comp_df["Alkane"].values,cw,color="#ed9e5c",alpha=0.85,label="Alkane",edgecolor="white",lw=0.5)
    ax.barh(cx+cw/2,comp_df["Non_alkane"].values,cw,color="#4a7ba7",alpha=0.85,label="Non-alkane",edgecolor="white",lw=0.5)
    ax.set_yticks(cx); ax.set_yticklabels(comp_df["Feature"].values,fontsize=11,fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (Test Set)",fontsize=14,fontweight="bold")
    ax.set_title("SHAP: Alkane vs Non-alkane",fontsize=15,fontweight="bold")
    ax.legend(fontsize=12,loc="lower right",framealpha=0.9); ax.tick_params(labelsize=12); label(ax,"(c)")

    # (d) 热图
    ax = fig.add_subplot(gs[1,1])
    top_heat=min(10,len(features)); top_samp=min(50,len(sv_test))
    top_names=df_imp.head(top_heat)["Feature"].tolist(); top_idx_h=[features.index(f) for f in top_names]
    shap_heat=sv_test[:top_samp,:][:,top_idx_h]; shap_heat_norm=shap_heat/(np.abs(shap_heat).max(axis=0,keepdims=True)+1e-9)
    im=ax.imshow(shap_heat_norm.T,aspect="auto",cmap=CMAP_BWOR,vmin=-1,vmax=1)
    ax.set_yticks(range(top_heat)); ax.set_yticklabels(top_names,fontsize=11,fontweight="bold")
    ax.set_xlabel(f"Test Sample Index (First {top_samp})",fontsize=14,fontweight="bold")
    ax.set_title("Normalized SHAP Heatmap",fontsize=15,fontweight="bold")
    ax.spines[:].set_visible(False); ax.tick_params(axis="x",length=0,labelsize=10); ax.tick_params(axis="y",length=0,labelsize=10)
    for ai in np.where(split_data["alk_test"][:top_samp])[0]: ax.axvline(ai,color=C_ORANGE,lw=1.2,alpha=0.6)
    cbar=plt.colorbar(im,ax=ax,fraction=0.025,pad=0.02); cbar.set_label("Normalized SHAP",fontsize=10,fontweight="bold"); cbar.ax.tick_params(labelsize=9)
    label(ax,"(d)")

    fig.savefig(RESULT_DIR/"Fig4_SHAP_Analysis.png"); plt.close(fig)
    print("[OK] Fig4_SHAP_Analysis.png")

# ---------- Fig.5 (适用域) ----------
def plot_fig5(best_model,best_name,split_data,preds_best):
    x_dev=np.vstack([split_data["X_train"],split_data["X_val"]]); y_dev=np.concatenate([split_data["y_train"],split_data["y_val"]])
    ltr,srtr,lte,srte,hstar=williams_ad(x_dev,y_dev,split_data["X_test"],split_data["y_test"],best_model)
    nd,nte=nn_dist(x_dev,split_data["X_test"]); dstar=np.percentile(nd,95)

    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1,2,figure=fig,wspace=0.32,left=0.07,right=0.96,top=0.92,bottom=0.13)

    # (a) Williams
    ax = fig.add_subplot(gs[0,0])
    mask_d_in=(ltr<=hstar)&(np.abs(srtr)<=3); mask_d_out=~mask_d_in
    mask_t_in=(lte<=hstar)&(np.abs(srte)<=3); mask_t_out=~mask_t_in
    n_t_out=mask_t_out.sum()

    ax.scatter(ltr[mask_d_in],srtr[mask_d_in],color=C_PRIMARY,s=18,alpha=0.40,edgecolors="none",
               label=f"Train+Val (in AD, $n$={mask_d_in.sum()})")
    ax.scatter(ltr[mask_d_out],srtr[mask_d_out],color=C_PRIMARY,s=45,alpha=0.95,marker="x",linewidths=2.0,
               label=f"Train+Val (out, $n$={mask_d_out.sum()})")
    ax.scatter(lte[mask_t_in],srte[mask_t_in],color=C_ORANGE,s=28,alpha=0.70,marker="s",edgecolors="none",
               label=f"Test (in AD, $n$={mask_t_in.sum()})")
    ax.scatter(lte[mask_t_out],srte[mask_t_out],color=C_RED,s=100,alpha=1.0,marker="s",edgecolors="none",zorder=10,
               label=f"Test (out, $n$={mask_t_out.sum()})")

    ax.axvline(hstar,color=C_RED,lw=2.2,ls="--"); ax.axhline(3,color=C_GRAY,lw=1.2,ls=":"); ax.axhline(-3,color=C_GRAY,lw=1.2,ls=":")
    ax.axvspan(0,hstar,color=C_LIGHT,alpha=0.06,zorder=0); ax.axhspan(-3,3,color=C_LIGHT,alpha=0.04,zorder=0)
    all_lev=np.concatenate([ltr,lte]); x_upper=max(float(np.nanmax(all_lev))*1.10,hstar*1.50)
    ax.set_xlim(0,x_upper); ax.set_ylim(-5.5,5.5)
    ax.set_xlabel("Leverage $h$",fontsize=14,fontweight="bold")
    ax.set_ylabel("Standardized Residual",fontsize=14,fontweight="bold")
    ax.set_title(f"Williams Plot ({best_name})",fontsize=15,fontweight="bold")
    ax.legend(fontsize=8.5,loc="upper right",framealpha=0.90,ncol=1,bbox_to_anchor=(1.0,0.98),borderaxespad=0)
    ax.tick_params(labelsize=12); label(ax,"(a)")
    print(f"  Williams AD: Test out = {n_t_out}")

    # (b) 距离密度 — 合并重复图例
    ax = fig.add_subplot(gs[0,1])
    bins=np.linspace(0,max(nd.max(),nte.max())*1.05,28)
    ax.hist(nd,bins=bins,color=C_PRIMARY,alpha=0.35,density=True,edgecolor="white",lw=0.3)
    ax.hist(nte,bins=bins,color=C_ORANGE,alpha=0.45,density=True,edgecolor="white",lw=0.3)
    ax.axvline(dstar,color=C_RED,lw=2.2,ls="--")
    n_out_d=(nte>dstar).sum()
    ax.text(0.97,0.95,
            f"Train+Val ($n$={len(nd)})  |  Test ($n$={len(nte)})\n$d^*$ (95th pctl) = {dstar:.2f}  |  Test beyond $d^*$: {n_out_d}/{len(nte)}",
            transform=ax.transAxes,ha="right",va="top",fontsize=12,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec=C_GRAY,alpha=0.92))
    ax.set_xlabel("Nearest-Neighbor Distance in Descriptor Space",fontsize=14,fontweight="bold")
    ax.set_ylabel("Probability Density",fontsize=14,fontweight="bold")
    ax.set_title("Distance to Development Domain",fontsize=15,fontweight="bold")
    ax.tick_params(labelsize=12); label(ax,"(b)")

    fig.savefig(RESULT_DIR/"Fig5_Applicability_Domain.png"); plt.close(fig)
    print("[OK] Fig5_Applicability_Domain.png")
    print(f"  Test in AD: {mask_t_in.sum()}, Test out: {n_t_out}, Dist out: {n_out_d}")

# ---------- Fig.6 (烷烃验证+MLR) ----------
def plot_fig6(group_metrics_all,split_data,preds_best,best_name):
    MLR_CANDIDATES=[Path(r"D:\res\6.8\MLR.xlsx"),Path(r"D:\Users\hkbg\Desktop\MLR.xlsx"),Path(__file__).resolve().parent/"MLR.xlsx"]
    mlr_file=None
    for p in MLR_CANDIDATES:
        if p.exists(): mlr_file=p; break
    has_mlr=mlr_file is not None

    y_test=split_data["y_test"]; y_pred=preds_best["test"]; alk_m=split_data["alk_test"]; nalk_m=~alk_m

    fig = plt.figure(figsize=(18,5.8))
    gs = gridspec.GridSpec(1,3,figure=fig,wspace=0.35,left=0.05,right=0.97,top=0.90,bottom=0.14)

    # (a) 雷达 — markersize减小, 标题对齐, 图例右下角
    ax = fig.add_subplot(gs[0,0],polar=True)
    angles=np.linspace(0,2*pi,3,endpoint=False).tolist()+[0]
    ax.set_theta_offset(pi/2); ax.set_theta_direction(-1); ax.set_rlim(0,1); ax.set_rticks([0.25,0.5,0.75,1.0])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(["$R^2$ (up)","1-RMSE* (up)","1-MAE* (up)"],fontsize=12,fontweight="bold")
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"],fontsize=9,fontweight="bold")

    all_r2,all_rmse,all_mae=[],[],[]
    for gm in group_metrics_all.values():
        for k in gm:
            if not np.isnan(gm[k]["R2"]): all_r2.append(gm[k]["R2"]); all_rmse.append(gm[k]["RMSE"]); all_mae.append(gm[k]["MAE"])
    r2_mn,r2_mx=min(all_r2),max(all_r2); rmse_mn,rmse_mx=min(all_rmse),max(all_rmse); mae_mn,mae_mx=min(all_mae),max(all_mae)

    gbm=group_metrics_all[best_name]
    for gn,st in [("Non-alkane",{"c":C_PRIMARY,"ls":"-","mk":"o"}),("Alkane",{"c":C_ORANGE,"ls":"--","mk":"^"})]:
        if not np.isnan(gbm[gn]["R2"]):
            m=gbm[gn]
            vs=[(m["R2"]-r2_mn)/max(r2_mx-r2_mn,1e-9),
                1-(m["RMSE"]-rmse_mn)/max(rmse_mx-rmse_mn,1e-9),
                1-(m["MAE"]-mae_mn)/max(mae_mx-mae_mn,1e-9)]
            vs+=vs[:1]
            ax.plot(angles,vs,color=st["c"],ls=st["ls"],marker=st["mk"],markersize=3.5,lw=2.8,label=f"{best_name} ({gn})")
            ax.fill(angles,vs,color=st["c"],alpha=0.06)
    ax.legend(loc="lower right",bbox_to_anchor=(1.15,-0.05),fontsize=11,framealpha=0.9)
    ax.set_title("GBDT: Alkane vs Non-alkane",pad=12,fontsize=14,fontweight="bold")
    # 缩小polar图顶部留白，使标题与(b)(c)对齐
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.92])
    label(ax,"(a)",x=-0.04,y=1.04)

    # (b) 分化学类别 parity — 删除右下角图例, 左上角文本框移到右下角
    ax = fig.add_subplot(gs[0,1])
    lo=min(y_test.min(),y_pred.min())-0.4; hi=max(y_test.max(),y_pred.max())+0.4
    ax.scatter(y_test[nalk_m],y_pred[nalk_m],color=C_PRIMARY,s=34,alpha=0.62,edgecolors="white",lw=0.3,
               label=f"Non-alkane ($n$={nalk_m.sum()})")
    ax.scatter(y_test[alk_m],y_pred[alk_m],color=C_ORANGE,marker="^",s=60,alpha=0.88,edgecolors="white",lw=0.5,
               label=f"Alkane ($n$={alk_m.sum()})")
    ax.plot([lo,hi],[lo,hi],color=C_RED,lw=2.0,ls="--",alpha=0.8)
    r2_a=r2_score(y_test[alk_m],y_pred[alk_m]) if alk_m.sum()>=2 else np.nan
    r2_n=r2_score(y_test[nalk_m],y_pred[nalk_m]) if nalk_m.sum()>=2 else np.nan
    rmse_a=np.sqrt(mean_squared_error(y_test[alk_m],y_pred[alk_m])); rmse_n=np.sqrt(mean_squared_error(y_test[nalk_m],y_pred[nalk_m]))
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel("Observed log$k$",fontsize=14,fontweight="bold")
    ax.set_ylabel("Predicted log$k$",fontsize=14,fontweight="bold")
    ax.set_title("Test Parity by Compound Type",fontsize=15,fontweight="bold")
    ax.legend(fontsize=12,loc="lower right",framealpha=0.9)
    ax.set_aspect("equal","box"); ax.tick_params(labelsize=12)
    label(ax,"(b)")

    # (c) MLR vs GBDT — 使用MLR完整烷烃数据(MAE=0.971), 删除图例
    ax = fig.add_subplot(gs[0,2])
    if has_mlr:
        names=split_data["names_test"]
        df_ours=pd.DataFrame({"Molecule":names[alk_m],"y_true":y_test[alk_m],"y_pred":y_pred[alk_m]})
        df_ours["bias"]=np.abs(df_ours["y_pred"]-df_ours["y_true"])

        # MLR: 使用MLR.xlsx中全部烷烃数据
        df_mlr=pd.read_excel(mlr_file); df_mlr["bias"]=np.abs(df_mlr["y_pred"]-df_mlr["y_true"])
        # MLR文件中所有烷烃
        b_mlr_all = df_mlr["bias"].values  # 全部17个
        m_mlr_all = b_mlr_all.mean()  # ~0.971
        b_ours = df_ours["bias"].values  # GBDT测试集烷烃9个
        m_ours = b_ours.mean()  # ~0.169
        impr=(m_mlr_all-m_ours)/(m_mlr_all+1e-9)*100

        # t-test on aligned subset
        df_merge=pd.merge(df_ours,df_mlr,on="Molecule",suffixes=("_ours","_mlr"))
        if len(df_merge)>5:
            _,p_val=stats.ttest_ind(df_merge["bias_ours"].values,df_merge["bias_mlr"].values,equal_var=False)
            ps=np.sqrt((np.var(df_merge["bias_ours"].values)+np.var(df_merge["bias_mlr"].values))/2)
            d_val=(df_merge["bias_mlr"].values.mean()-df_merge["bias_ours"].values.mean())/(ps+1e-9)
        else:
            p_val=float('nan'); d_val=float('nan')

        # 小提琴 + 箱线 + 散点 (MLR全部, GBDT测试集)
        parts=ax.violinplot([b_mlr_all,b_ours],positions=[0,1],widths=0.60,showextrema=False)
        for body,col in zip(parts["bodies"],[C_ORANGE,C_PRIMARY]):
            body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.28)
        ax.boxplot([b_mlr_all,b_ours],positions=[0,1],widths=0.16,patch_artist=True,
                   boxprops=dict(facecolor="white",lw=1.5),medianprops=dict(color=C_RED,lw=2.8),
                   whiskerprops=dict(lw=1.3),capprops=dict(lw=1.3))
        rng=np.random.default_rng(42)
        for xp,vals,col in zip([0,1],[b_mlr_all,b_ours],[C_ORANGE,C_PRIMARY]):
            j=rng.uniform(-0.10,0.10,len(vals))
            ax.scatter(np.full(len(vals),xp)+j,vals,color=col,s=30,alpha=0.60,edgecolors="white",lw=0.3)
        ax.errorbar([0,1],[m_mlr_all,m_ours],fmt='D',color='black',capsize=5,ms=9,
                    markeredgecolor='white',markeredgewidth=1.2,zorder=10)
        # 统计文字: 白色框, 右上角
        ax.text(0.97,0.94,
                f"MLR MAE = {m_mlr_all:.4f}  |  GBDT MAE = {m_ours:.4f}\nReduction = {impr:.1f}%  |  p = {p_val:.2e}",
                transform=ax.transAxes,ha="right",va="top",fontsize=11,fontweight="bold",color=C_PRIMARY,
                bbox=dict(boxstyle="round,pad=0.4",fc="white",ec=C_GRAY,alpha=0.92))
        ax.set_xticks([0,1]); ax.set_xticklabels(["MLR\n(Prior Work)","GBDT\n(This Study)"],fontsize=12,fontweight="bold")
        ax.axhline(0,color=C_RED,lw=1.5,ls="--",alpha=0.5)
    else:
        ax.text(0.5,0.5,"MLR data not available",ha="center",va="center",transform=ax.transAxes,fontsize=14,fontweight="bold",color=C_GRAY)
    ax.set_ylabel("Absolute Prediction Error |Pred - Obs|",fontsize=14,fontweight="bold")
    ax.set_title("Alkane Bias: MLR vs GBDT",fontsize=15,fontweight="bold")
    ax.tick_params(labelsize=12); label(ax,"(c)")

    fig.savefig(RESULT_DIR/"Fig6_Alkane_MLR_Combined.png"); plt.close(fig)
    print("[OK] Fig6_Alkane_MLR_Combined.png")
    if has_mlr: print(f"  GBDT MAE={m_ours:.4f}, MLR MAE={m_mlr_all:.4f}, Impr={impr:.1f}%, p={p_val:.2e}, d={d_val:.2f}")

# ============================================================
# 4. 主程序
# ============================================================
def main():
    print("="*60+"\n  Paper Figures v2.4\n"+"="*60)
    excel_file=resolve_excel_file(); print(f"\nData: {excel_file}")
    data,features,system_frames=load_data(excel_file); sd=split_scale(data,features); sd["features"]=features
    print(f"N={len(data)}, P={len(features)}")

    print("\nTraining..."); models,best_name,best_model,metrics_val,preds_val,gm_all,preds_best=train_all(sd)
    print(f"Best: {best_name}")

    # ============== 终端输出所有模型全部结果 ==============
    print(f"\n{'='*70}")
    print(f"  ALL MODELS — Train / Val / Test Results")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'Split':<8} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mn in ["Ridge","SVR","RF","GBDT"]:
        m = models[mn]
        for lbl, Xs, ys in [("Train", sd["X_train"], sd["y_train"]),
                             ("Val",   sd["X_val"],   sd["y_val"]),
                             ("Test",  sd["X_test"],  sd["y_test"])]:
            yp = m.predict(Xs)
            r2_ = r2_score(ys, yp)
            rmse_ = np.sqrt(mean_squared_error(ys, yp))
            mae_ = mean_absolute_error(ys, yp)
            mark = "  <<< BEST" if (mn==best_name and lbl=="Val") else ""
            print(f"  {mn:<8} {lbl:<8} {r2_:>8.4f} {rmse_:>8.4f} {mae_:>8.4f}{mark}")

    # ============== 测试集分烷烃/非烷烃 ==============
    print(f"\n{'='*70}")
    print(f"  TEST SET — Alkane vs Non-alkane (All Models)")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'Group':<12} {'n':>4} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
    for mn in ["Ridge","SVR","RF","GBDT"]:
        m = models[mn]
        yp_test = m.predict(sd["X_test"])
        for gn, mask in [("Non-alkane", ~sd["alk_test"]), ("Alkane", sd["alk_test"])]:
            if mask.sum() >= 2:
                r2_  = r2_score(sd["y_test"][mask], yp_test[mask])
                rmse_ = np.sqrt(mean_squared_error(sd["y_test"][mask], yp_test[mask]))
                mae_ = mean_absolute_error(sd["y_test"][mask], yp_test[mask])
                print(f"  {mn:<8} {gn:<12} {mask.sum():>4} {r2_:>8.4f} {rmse_:>8.4f} {mae_:>8.4f}")
            else:
                print(f"  {mn:<8} {gn:<12} {mask.sum():>4} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    print(f"{'='*70}")

    print("SHAP..."); explainer=shap.TreeExplainer(best_model)
    sv_tr=explainer(sd["X_train"]).values; sv_te=explainer(sd["X_test"]).values
    shap_imp=np.abs(sv_tr).mean(0)
    df_imp=pd.DataFrame({"Feature":features,"Importance":shap_imp}).sort_values("Importance",ascending=False)
    df_comp=pd.DataFrame({"Feature":features,"Alkane":np.abs(sv_te[sd["alk_test"]]).mean(0),
                           "Non_alkane":np.abs(sv_te[~sd["alk_test"]]).mean(0)})
    df_comp["Diff"]=df_comp["Alkane"]-df_comp["Non_alkane"]; df_comp=df_comp.sort_values("Diff",ascending=False)

    print("ACP..."); cr,cs=fit_acp(sd,ns=5,beta=1.0,alphas=(0.10,0.05))

    print("\n"+"="*40+"\nRendering...\n"+"="*40)
    plot_fig1(data,sd,system_frames)
    plot_fig2(metrics_val,sd,preds_val,preds_best,best_name)
    plot_fig3(cr,cs)
    plot_fig4(best_model,sd,df_imp,df_comp)
    plot_fig5(best_model,best_name,sd,preds_best)
    plot_fig6(gm_all,sd,preds_best,best_name)
    print(f"\n{'='*60}\n  Done! -> {RESULT_DIR}\n{'='*60}")

if __name__=="__main__": main()
