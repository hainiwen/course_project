"""Build 04_msm_levofloxacin.ipynb"""
import json, textwrap

OUT_NB = "c:/ARMD/Python_Coursework_ARMD_analysis/python_script/04_msm_levofloxacin.ipynb"
OUT_DIR = "c:/ARMD/Python_Coursework_ARMD_analysis/python_output/"

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip()}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Notebook 04 — Multi-State Model (E. coli Levofloxacin)

**2-state continuous-time Markov chain:** Susceptible (S) ↔ notS (Intermediate or Resistant)

**Approach:** Custom MLE via `scipy.linalg.expm` + `scipy.optimize.minimize`

**Sections:**
1. Setup & Load
2. Data Preparation
3. Q Matrix & Likelihood Engine
4. Forward Stepwise Model Selection (candidates: nh_90d, fq_90d, outpatient)
5. AIC Comparison
6. Hazard Ratios
7. Sojourn Profiles
8. State Occupancy Plot
9. Diagnostics — Prevalence plot, KM vs predicted, Obs vs expected transition counts
10. HR Forest Plot
"""))

# ── Section 1: Setup ───────────────────────────────────────────────────────────
cells.append(md("## Section 1 — Setup & Load"))

cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import expm
from scipy.optimize import minimize
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = "c:/ARMD/Python_Coursework_ARMD_analysis/python_output/"
COV_FILE = "c:/ARMD/output6/03_ecoli_lev_cov_dataset.csv"

cov = pd.read_csv(COV_FILE)
cov["order_time"] = pd.to_datetime(cov["order_time"])

# Fill NA in binary covariates with 0
for col in ["fq_90d", "outpatient", "nh_90d"]:
    cov[col] = cov[col].fillna(0).astype(int)

print(f"Rows: {len(cov):,}  |  Patients: {cov['anon_id'].nunique():,}")
print("\\nSusceptibility distribution:")
print(cov["susceptibility"].value_counts())
cov.head(3)
"""))

# ── Section 2: Data Prep ───────────────────────────────────────────────────────
cells.append(md("## Section 2 — Data Preparation for MSM"))

cells.append(code("""\
# State encoding: 1 = Susceptible, 2 = notS (I or R)
cov["state"] = np.where(cov["susceptibility"] == "Susceptible", 1, 2)

# Time in years from each patient's first observation
cov["t_yr"] = cov.groupby("anon_id")["order_time"].transform(
    lambda x: (x - x.min()).dt.days / 365.25
)

# Sort
cov = cov.sort_values(["anon_id", "t_yr"]).reset_index(drop=True)

# Remove exact (anon_id, t_yr) duplicates — keep first
cov = cov.drop_duplicates(subset=["anon_id", "t_yr"], keep="first")

# Build consecutive transition pairs
msm = cov.copy()
msm["state_next"] = msm.groupby("anon_id")["state"].shift(-1)
msm["t_next"]     = msm.groupby("anon_id")["t_yr"].shift(-1)
msm["dt"]         = msm["t_next"] - msm["t_yr"]

# Drop last obs per patient and zero-dt rows
msm = msm.dropna(subset=["state_next", "dt"])
msm = msm[msm["dt"] > 0].copy()
msm["state_next"] = msm["state_next"].astype(int)

print(f"Transition pairs: {len(msm):,}")
print("\\nObserved transition counts:")
tc = msm.groupby(["state","state_next"]).size().unstack(fill_value=0)
tc.index   = ["S(1)", "notS(2)"]
tc.columns = ["to S(1)", "to notS(2)"]
print(tc)

# Proportions
tc_prop = tc.div(tc.sum(axis=1), axis=0).round(3)
print("\\nRow proportions:")
print(tc_prop)

# Save
msm[["anon_id","t_yr","state","state_next","dt",
     "fq_90d","outpatient","nh_90d","adi_score"]].to_csv(
    OUT_DIR + "04_ecoli_lev_msm_data.csv", index=False)
print("\\nSaved: 04_ecoli_lev_msm_data.csv")
"""))

# ── Section 3: Likelihood engine ───────────────────────────────────────────────
cells.append(md("""\
## Section 3 — Q Matrix & Log-Likelihood Engine

For a 2-state CTMC with covariates, the intensity matrix at observation $i$ is:
$$Q_i = \\begin{pmatrix} -q_{12}^{(i)} & q_{12}^{(i)} \\\\ q_{21}^{(i)} & -q_{21}^{(i)} \\end{pmatrix}$$
where $q_{12}^{(i)} = q_{12} \\cdot e^{\\boldsymbol{\\beta}_{12} \\cdot \\mathbf{x}_i}$

Transition probabilities: $P(\\Delta t) = e^{Q_i \\cdot \\Delta t}$ (matrix exponential via `scipy.linalg.expm`)
"""))

cells.append(code("""\
# Initial Q matrix (per year) — from observed empirical transition rates
Q_INIT = np.array([[-0.1618,  0.1618],
                   [ 0.2711, -0.2711]])

print("Initial Q matrix (per year):")
print(pd.DataFrame(Q_INIT, index=["S","notS"], columns=["to S","to notS"]).round(4))
print(f"\\nImplied sojourn in S:    {1/0.1618*365.25:.0f} days")
print(f"Implied sojourn in notS: {1/0.2711*365.25:.0f} days")
"""))

cells.append(code("""\
# Negative log-likelihood (vectorised over observations, loop over expm calls)
# params layout: [log_q12, log_q21,
#                 beta_q12_cov1, beta_q21_cov1,
#                 beta_q12_cov2, beta_q21_cov2, ...]

def neg_log_lik(params, dt_arr, from_arr, to_arr, cov_arrays):
    q12_base = np.exp(params[0])
    q21_base = np.exp(params[1])

    # Accumulate linear predictor per transition type
    lp12 = np.zeros(len(dt_arr))
    lp21 = np.zeros(len(dt_arr))
    for j, x in enumerate(cov_arrays):
        lp12 += params[2 + 2*j]     * x
        lp21 += params[2 + 2*j + 1] * x

    q12_i = q12_base * np.exp(lp12)
    q21_i = q21_base * np.exp(lp21)

    ll = 0.0
    for i in range(len(dt_arr)):
        Q = np.array([[-q12_i[i],  q12_i[i]],
                      [ q21_i[i], -q21_i[i]]])
        P = expm(Q * dt_arr[i])
        p = P[from_arr[i]-1, to_arr[i]-1]
        p = max(float(p), 1e-300)   # numerical floor
        ll += np.log(p)
    return -ll


# Pre-extract numpy arrays (speed)
dt_arr   = msm["dt"].values
from_arr = msm["state"].values.astype(int)
to_arr   = msm["state_next"].values.astype(int)

print(f"Likelihood engine ready — {len(dt_arr):,} transition pairs")

# Quick sanity check on M0 starting point
x0_m0 = np.array([np.log(0.1618), np.log(0.2711)])
ll_init = -neg_log_lik(x0_m0, dt_arr, from_arr, to_arr, [])
print(f"Log-likelihood at Q_INIT: {ll_init:.2f}")
"""))

# ── Section 4: Forward stepwise model fitting ──────────────────────────────────
cells.append(md("""\
## Section 4 — Forward Stepwise Model Selection

Candidates (from univariable analysis in NB02): `nh_90d`, `fq_90d`, `outpatient`.

**Strategy:**
1. Fit M0 (baseline, no covariates).
2. Step 1 — try each candidate alone; keep the one with the biggest AIC drop (threshold: ΔAIC < −4).
3. Step 2 — try adding each remaining candidate to the Step-1 winner; keep the best if ΔAIC < −4.
4. Step 3 — try adding the last candidate to the Step-2 model; keep if ΔAIC < −4.
5. Stop as soon as no further meaningful improvement.

**Threshold:** ΔAIC < −4 (Burnham & Anderson convention for meaningful improvement).

**Optimiser strategy:** Nelder-Mead first (robust to expm overflow) → warm L-BFGS-B from NM solution.
"""))

cells.append(code("""\
def fit_model(cov_names, label=""):
    cov_arrays = [msm[c].values.astype(float) for c in cov_names]
    n_p = 2 + 2 * len(cov_names)
    x0  = np.zeros(n_p)
    x0[0] = np.log(0.1618)
    x0[1] = np.log(0.2711)

    args = (dt_arr, from_arr, to_arr, cov_arrays)

    # Step 1: Nelder-Mead (robust)
    res_nm = minimize(neg_log_lik, x0, args=args, method="Nelder-Mead",
                      options={"maxiter":20000, "xatol":1e-6, "fatol":1e-6})

    # Step 2: warm L-BFGS-B from NM solution
    res_bfgs = minimize(neg_log_lik, res_nm.x, args=args, method="L-BFGS-B",
                        options={"maxiter":2000})

    best = res_bfgs if res_bfgs.fun < res_nm.fun else res_nm
    loglik = -best.fun
    aic    = -2*loglik + 2*n_p

    if label:
        print(f"  {label}: AIC={aic:.2f}  logLik={loglik:.2f}  converged={best.success}  "
              f"q12={np.exp(best.x[0]):.4f}  q21={np.exp(best.x[1]):.4f}")
    return {"label":label, "cov_names":list(cov_names), "params":best.x,
            "logLik":loglik, "AIC":aic, "converged":best.success, "n_param":n_p}


CANDIDATES = ["nh_90d", "fq_90d", "outpatient"]

print("Fitting M0 (baseline)...")
m0 = fit_model([], "M0 baseline")
all_models = [m0]
"""))

cells.append(code("""\
# ── Step 1: each candidate individually ────────────────────────────────────────
print("\\n--- Step 1: try each candidate alone ---")
step1 = {}
for cand in CANDIDATES:
    r = fit_model([cand], f"M1_{cand}")
    step1[cand] = r
    print(f"    DELTA-AIC vs M0: {r['AIC'] - m0['AIC']:+.2f}")
    all_models.append(r)

best_s1_name = min(step1, key=lambda c: step1[c]["AIC"])
m1_best = step1[best_s1_name]
print(f"\\nStep-1 winner: {best_s1_name}  (AIC={m1_best['AIC']:.2f},  "
      f"dAIC vs M0={m1_best['AIC']-m0['AIC']:+.2f})")

AIC_THRESHOLD = -4   # DELTA-AIC must be < -4 to add a covariate (Burnham & Anderson)

if m1_best["AIC"] - m0["AIC"] >= AIC_THRESHOLD:
    print(f"No meaningful improvement over M0 (threshold DELTA-AIC < {AIC_THRESHOLD}) — stopping at baseline.")
    best_model = m0
else:
    # ── Step 2: add each remaining candidate to step-1 winner ─────────────────
    remaining1 = [c for c in CANDIDATES if c != best_s1_name]
    print(f"\\n--- Step 2: add each remaining candidate to {best_s1_name} ---")
    step2 = {}
    for cand in remaining1:
        covs2 = m1_best["cov_names"] + [cand]
        r = fit_model(covs2, f"M2_{best_s1_name}+{cand}")
        step2[cand] = r
        print(f"    DELTA-AIC vs step-1: {r['AIC'] - m1_best['AIC']:+.2f}")
        all_models.append(r)

    best_s2_name = min(step2, key=lambda c: step2[c]["AIC"])
    m2_best = step2[best_s2_name]
    print(f"\\nStep-2 winner: +{best_s2_name}  (AIC={m2_best['AIC']:.2f},  "
          f"dAIC vs step-1={m2_best['AIC']-m1_best['AIC']:+.2f})")

    if m2_best["AIC"] - m1_best["AIC"] >= AIC_THRESHOLD:
        print(f"No meaningful improvement at step 2 (threshold DELTA-AIC < {AIC_THRESHOLD}) — stopping.")
        best_model = m1_best
    else:
        # ── Step 3: try the last remaining candidate ───────────────────────────
        remaining2 = [c for c in remaining1 if c != best_s2_name]
        if remaining2:
            last_cov = remaining2[0]
            covs3 = m2_best["cov_names"] + [last_cov]
            print(f"\\n--- Step 3: try adding {last_cov} to {m2_best['label']} ---")
            m3_try = fit_model(covs3, f"M3_all3")
            print(f"    DELTA-AIC vs step-2: {m3_try['AIC'] - m2_best['AIC']:+.2f}")
            all_models.append(m3_try)
            if m3_try["AIC"] - m2_best["AIC"] < AIC_THRESHOLD:
                print(f"  -> Meaningful improvement (DELTA-AIC < {AIC_THRESHOLD}) — adding {last_cov}.")
                best_model = m3_try
            else:
                print(f"  -> No meaningful improvement — stopping at two-variable model.")
                best_model = m2_best
        else:
            best_model = m2_best

print(f"\\n==> Final selected model: {best_model['label']}")
print(f"    Covariates: {best_model['cov_names']}")
print(f"    AIC: {best_model['AIC']:.2f}  |  logLik: {best_model['logLik']:.2f}")
"""))

# ── Section 5: AIC ─────────────────────────────────────────────────────────────
cells.append(md("## Section 5 — AIC Comparison"))

cells.append(code("""\
aic_df = pd.DataFrame([{
    "model":     m["label"],
    "covariates": ", ".join(m["cov_names"]) if m["cov_names"] else "(none)",
    "n_param":   m["n_param"],
    "logLik":    round(m["logLik"], 2),
    "AIC":       round(m["AIC"], 3),
    "converged": m["converged"]
} for m in all_models])
aic_df["dAIC"] = (aic_df["AIC"] - aic_df["AIC"].min()).round(3)
aic_df = aic_df.sort_values("AIC").reset_index(drop=True)

print(aic_df.to_string(index=False))
print(f"\\nSelected model: {best_model['label']}")

aic_df.to_csv(OUT_DIR + "04_ecoli_lev_aic_table.csv", index=False)
print("Saved: 04_ecoli_lev_aic_table.csv")
"""))

# ── Section 6: HRs ─────────────────────────────────────────────────────────────
cells.append(md("""\
## Section 6 — Hazard Ratios from Best Model

HR = exp(β). CIs from numerical Hessian (if positive definite), else reported as NA.

Each covariate has **two betas**: one for S→notS (q12) and one for notS→S (q21).
"""))

cells.append(code("""\
def extract_hrs(model):
    params    = model["params"]
    cov_names = model["cov_names"]
    n_cov     = len(cov_names)

    # Numerical Hessian for CIs
    from scipy.optimize import approx_fprime
    cov_arrays = [msm[c].values.astype(float) for c in cov_names]
    args = (dt_arr, from_arr, to_arr, cov_arrays)

    eps = 1e-5
    grad_f = lambda p: approx_fprime(p, neg_log_lik, eps, *args)
    try:
        H = np.zeros((len(params), len(params)))
        for i in range(len(params)):
            e = np.zeros(len(params)); e[i] = eps
            H[i] = (grad_f(params + e) - grad_f(params - e)) / (2*eps)
        H = (H + H.T) / 2   # symmetrise
        cov_mat = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov_mat))
        hess_ok = np.all(np.diag(cov_mat) > 0)
    except Exception:
        se = np.full(len(params), np.nan)
        hess_ok = False

    rows = []
    for j, cname in enumerate(cov_names):
        for trans, idx, label in [("S→notS", 2+2*j, "q12"), ("notS→S", 2+2*j+1, "q21")]:
            beta = params[idx]
            hr   = np.exp(beta)
            ci_lo = np.exp(beta - 1.96*se[idx]) if hess_ok else np.nan
            ci_hi = np.exp(beta + 1.96*se[idx]) if hess_ok else np.nan
            rows.append({"covariate":cname, "transition":trans,
                         "beta":round(beta,4), "HR":round(hr,3),
                         "CI_lo":round(ci_lo,3) if not np.isnan(ci_lo) else None,
                         "CI_hi":round(ci_hi,3) if not np.isnan(ci_hi) else None})
    note = "" if hess_ok else "  [Hessian not PD — CIs not available]"
    return pd.DataFrame(rows), note

hr_df, hr_note = extract_hrs(best_model)
print(f"HRs from {best_model['label']}{hr_note}")
print(hr_df.to_string(index=False))

hr_df.to_csv(OUT_DIR + "04_ecoli_lev_hr_table.csv", index=False)
print("\\nSaved: 04_ecoli_lev_hr_table.csv")
"""))

# ── Section 7: Sojourn ─────────────────────────────────────────────────────────
cells.append(md("""\
## Section 7 — Sojourn Time Profiles

Sojourn in state S = 1/q12, in state notS = 1/q21 (years → × 365.25 for days).

Compare reference (all covariates = 0) vs each covariate exposed (= 1).
"""))

cells.append(code("""\
def sojourn_profile(model):
    params    = model["params"]
    cov_names = model["cov_names"]
    q12_base  = np.exp(params[0])
    q21_base  = np.exp(params[1])

    rows = [{"profile":"Reference (all=0)",
             "sojourn_S_days":   round(365.25/q12_base, 1),
             "sojourn_notS_days":round(365.25/q21_base, 1)}]

    for j, cname in enumerate(cov_names):
        b12 = params[2 + 2*j]
        b21 = params[2 + 2*j + 1]
        q12_exp = q12_base * np.exp(b12)
        q21_exp = q21_base * np.exp(b21)
        rows.append({"profile": f"{cname}=1 (others=0)",
                     "sojourn_S_days":   round(365.25/q12_exp, 1),
                     "sojourn_notS_days":round(365.25/q21_exp, 1)})
    return pd.DataFrame(rows)

soj_df = sojourn_profile(best_model)
print(f"Sojourn profiles ({best_model['label']}):")
print(soj_df.to_string(index=False))

soj_df.to_csv(OUT_DIR + "04_ecoli_lev_sojourn_profiles.csv", index=False)
print("\\nSaved: 04_ecoli_lev_sojourn_profiles.csv")
"""))

# ── Section 8: State occupancy ─────────────────────────────────────────────────
cells.append(md("""\
## Section 8 — State Occupancy Plot

Starting from state S, predict P(in S at time t) and P(in notS at time t)
using $P(t) = e^{Q \\cdot t}$ for reference vs fq-exposed.
"""))

cells.append(code("""\
def state_occupancy(model, cov_vals=None):
    params   = model["params"]
    cov_names = model["cov_names"]
    q12_base = np.exp(params[0])
    q21_base = np.exp(params[1])

    lp12 = lp21 = 0.0
    if cov_vals:
        for j, cname in enumerate(cov_names):
            x = cov_vals.get(cname, 0)
            lp12 += params[2 + 2*j]     * x
            lp21 += params[2 + 2*j + 1] * x

    q12 = q12_base * np.exp(lp12)
    q21 = q21_base * np.exp(lp21)
    Q   = np.array([[-q12, q12], [q21, -q21]])

    t_grid = np.linspace(0, 5, 300)   # 0–5 years
    p0     = np.array([1.0, 0.0])     # start in S
    pS     = np.array([(p0 @ expm(Q * t))[0] for t in t_grid])
    return t_grid * 365.25, pS        # return time in days


t_days_ref, pS_ref = state_occupancy(best_model, cov_vals={c:0 for c in best_model["cov_names"]})

# Pick fq_90d for contrast (if in model)
if "fq_90d" in best_model["cov_names"]:
    fq_vals = {c:0 for c in best_model["cov_names"]}
    fq_vals["fq_90d"] = 1
    t_days_fq, pS_fq = state_occupancy(best_model, cov_vals=fq_vals)
else:
    t_days_fq, pS_fq = None, None

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(t_days_ref, pS_ref,     color="steelblue", lw=2, label="P(S) — reference")
ax.plot(t_days_ref, 1-pS_ref,   color="steelblue", lw=2, ls="--", label="P(notS) — reference")
if pS_fq is not None:
    ax.plot(t_days_fq, pS_fq,   color="coral", lw=2, label="P(S) — fq_90d=1")
    ax.plot(t_days_fq, 1-pS_fq, color="coral", lw=2, ls="--", label="P(notS) — fq_90d=1")

ax.set_xlabel("Days from first observation")
ax.set_ylabel("Predicted probability")
ax.set_title(f"State Occupancy — {best_model['label']}")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR + "04_ecoli_lev_state_occupancy.png", dpi=150)
plt.show()
print("Saved: 04_ecoli_lev_state_occupancy.png")
"""))

# ── Section 9: Diagnostics ─────────────────────────────────────────────────────
cells.append(md("""\
## Section 9 — Model Diagnostics

Three complementary checks:

### 9a. Prevalence Plot
Bin observations by time → compare observed % in state S vs model-predicted.
Equivalent to R's `plot.msm()`. Tells you if the model tracks marginal state distribution over time.

### 9b. KM vs Model-Predicted Survival (KM VPC)
For S→notS: the CTMC predicts exponential survival P(still S at t) = exp(−q12·t).
Overlay against empirical KM. Divergence = exponential sojourn assumption violated.

### 9c. Observed vs Expected Transition Counts
Bin data by time interval length → compare observed transition counts to model-expected.
"""))

cells.append(code("""\
# Restore cov DataFrame (shadowed by loop variable in Section 4)
cov = pd.read_csv(COV_FILE)
cov["order_time"] = pd.to_datetime(cov["order_time"])
for col in ["fq_90d", "outpatient", "nh_90d"]:
    cov[col] = cov[col].fillna(0).astype(int)
cov["state"] = np.where(cov["susceptibility"] == "Susceptible", 1, 2)
cov["t_yr"] = cov.groupby("anon_id")["order_time"].transform(
    lambda x: (x - x.min()).dt.days / 365.25
)
cov = cov.sort_values(["anon_id", "t_yr"]).reset_index(drop=True)
cov = cov.drop_duplicates(subset=["anon_id", "t_yr"], keep="first")
print(f"cov restored: {len(cov):,} rows")
"""))

cells.append(code("""\
# ── 9a: Prevalence Plot ──────────────────────────────────────────────────────

# Bin observations by t_yr into equal-width bins
n_bins = 12
cov["t_bin"] = pd.cut(cov["t_yr"], bins=n_bins)
obs_prev = cov.groupby("t_bin", observed=True)["state"].apply(
    lambda x: (x == 1).mean()
).reset_index()
obs_prev.columns = ["t_bin", "obs_pS"]
obs_prev["t_mid"] = obs_prev["t_bin"].apply(lambda x: x.mid)

# Model-predicted P(S) at each bin midpoint (reference profile)
params   = best_model["params"]
q12_base = np.exp(params[0])
q21_base = np.exp(params[1])
Q_ref    = np.array([[-q12_base, q12_base], [q21_base, -q21_base]])
p0       = np.array([1.0, 0.0])

# Start from p0=[1,0]: index-S cohort all start in S
p0 = np.array([1.0, 0.0])
obs_prev["pred_pS"] = obs_prev["t_mid"].apply(
    lambda t: float((p0 @ expm(Q_ref * t))[0])
)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(obs_prev["t_mid"], obs_prev["obs_pS"],  "o-", color="steelblue", label="Observed % in S")
ax.plot(obs_prev["t_mid"], obs_prev["pred_pS"], "s--", color="coral",    label="Model-predicted % in S")
ax.set_xlabel("Years from first observation")
ax.set_ylabel("Proportion in Susceptible state")
ax.set_title("Diagnostic 9a: Prevalence Plot")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR + "04_ecoli_lev_diag_prevalence.png", dpi=150)
plt.show()
print("Saved: 04_ecoli_lev_diag_prevalence.png")
"""))

cells.append(code("""\
# ── 9b: Observed vs Model-Predicted P(S→S) by interval length ────────────────
# Uses msm rows (state==1) — same data the model was fitted on.
# For each row, compute predicted P[1,1](dt) using that row's own covariates.
# Then bin by dt and compare observed vs predicted fraction staying in S.

s_rows = msm[msm["state"] == 1].copy()
s_rows["event"] = (s_rows["state_next"] == 2).astype(int)

def p11_row(row):
    lp12 = sum(best_model["params"][2+2*j] * row[c]
               for j, c in enumerate(best_model["cov_names"]))
    lp21 = sum(best_model["params"][2+2*j+1] * row[c]
               for j, c in enumerate(best_model["cov_names"]))
    q12_i = q12_base * np.exp(lp12)
    q21_i = q21_base * np.exp(lp21)
    P = expm(np.array([[-q12_i, q12_i],[q21_i,-q21_i]]) * row["dt"])
    return float(P[0, 0])

s_rows["pred_p11"] = s_rows.apply(p11_row, axis=1)

dt_bins   = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf]
dt_labels = ["<0.1","0.1-0.25","0.25-0.5","0.5-1","1-2","2-5",">5"]
s_rows["dt_bin"] = pd.cut(s_rows["dt"], bins=dt_bins, labels=dt_labels)

vpc = s_rows.groupby("dt_bin", observed=True).agg(
    n=("event","count"),
    obs_pSS=("event", lambda x: 1 - x.mean()),
    pred_pSS=("pred_p11","mean")
).reset_index()
vpc = vpc[vpc["n"] >= 10]

fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(vpc))
ax.plot(x, vpc["obs_pSS"],  "o-", color="steelblue", label="Observed P(S→S)")
ax.plot(x, vpc["pred_pSS"], "s--", color="coral",    label="Model-predicted P(S→S)")
ax.set_xticks(x)
ax.set_xticklabels(vpc["dt_bin"].astype(str) + "\\nn=" + vpc["n"].astype(str), fontsize=8)
ax.set_xlabel("Interval length (years)")
ax.set_ylabel("Proportion staying in S")
ax.set_title("Diagnostic 9b: Observed vs Model-Predicted P(S→S) by interval length")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR + "04_ecoli_lev_diag_km_vpc.png", dpi=150)
plt.show()
print(vpc[["dt_bin","n","obs_pSS","pred_pSS"]].round(3).to_string(index=False))
print("\\nSaved: 04_ecoli_lev_diag_km_vpc.png")
"""))

cells.append(code("""\
# ── 9c: Observed vs Expected Transition Counts by dt Bin ─────────────────────

# Bin transition pairs by dt (time gap in years)
dt_bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
dt_labels = ["0-0.1yr","0.1-0.25yr","0.25-0.5yr","0.5-1yr","1-2yr",">2yr"]
msm["dt_bin"] = pd.cut(msm["dt"], bins=dt_bins, labels=dt_labels)

rows = []
for bin_label, grp in msm.groupby("dt_bin", observed=True):
    if len(grp) == 0:
        continue
    dt_mid = grp["dt"].median()
    for fs in [1,2]:
        for ts in [1,2]:
            obs = ((grp["state"]==fs) & (grp["state_next"]==ts)).sum()
            if obs == 0:
                continue
            # Expected: sum of P[fs,ts] for each row
            cov_arrays_m = [grp[c].values.astype(float) for c in best_model["cov_names"]]
            exp_count = 0.0
            for i, (_, row) in enumerate(grp.iterrows()):
                lp12_r = sum(best_model["params"][2+2*j] * row[c]
                             for j,c in enumerate(best_model["cov_names"]))
                lp21_r = sum(best_model["params"][2+2*j+1] * row[c]
                             for j,c in enumerate(best_model["cov_names"]))
                q12_r = q12_base * np.exp(lp12_r)
                q21_r = q21_base * np.exp(lp21_r)
                Qr    = np.array([[-q12_r, q12_r],[q21_r,-q21_r]])
                Pr    = expm(Qr * row["dt"])
                exp_count += Pr[fs-1, ts-1]
            trans_lbl = f"{'S' if fs==1 else 'notS'} -> {'S' if ts==1 else 'notS'}"
            rows.append({"dt_bin":bin_label,"transition":trans_lbl,
                         "observed":obs,"expected":round(exp_count,1),
                         "ratio":round(obs/exp_count,3) if exp_count>0 else None})

obs_exp_df = pd.DataFrame(rows)
print("Observed vs Expected transition counts by dt bin:")
print(obs_exp_df.to_string(index=False))
obs_exp_df.to_csv(OUT_DIR + "04_ecoli_lev_diag_obs_exp.csv", index=False)
print("\\nSaved: 04_ecoli_lev_diag_obs_exp.csv")
"""))

cells.append(code("""\
# ── 9c plot: grouped bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14,4), sharey=False)
trans_labels = ["S -> S","S -> notS","notS -> S","notS -> notS"]

for ax, tlabel in zip(axes, trans_labels):
    sub = obs_exp_df[obs_exp_df["transition"]==tlabel]
    if sub.empty:
        ax.set_title(tlabel); ax.axis("off"); continue
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w/2, sub["observed"], w, color="steelblue", label="Observed")
    ax.bar(x + w/2, sub["expected"], w, color="coral",     label="Expected", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["dt_bin"], rotation=30, ha="right", fontsize=7)
    ax.set_title(tlabel, fontsize=9)
    ax.legend(fontsize=7)

fig.suptitle("Diagnostic 9c: Observed vs Expected Transitions by dt Bin", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR + "04_ecoli_lev_diag_obs_exp.png", dpi=150)
plt.show()
print("Saved: 04_ecoli_lev_diag_obs_exp.png")
"""))

# ── Section 10: HR Forest Plot ──────────────────────────────────────────────────
cells.append(md("## Section 10 — HR Forest Plot\n\nHazard ratios from the best model, for both S→notS and notS→S transitions."))

cells.append(code("""\
# Split by transition
hr_s_nots = hr_df[hr_df["transition"] == "S\u2192notS"].copy()
hr_nots_s = hr_df[hr_df["transition"] == "notS\u2192S"].copy()

fig, axes = plt.subplots(1, 2, figsize=(11, max(3, len(hr_df)//2 + 1)), sharey=False)

for ax, sub, title, color in [
    (axes[0], hr_s_nots, "S \u2192 notS (Resistance acquisition)", "coral"),
    (axes[1], hr_nots_s, "notS \u2192 S (Susceptibility recovery)", "steelblue"),
]:
    if sub.empty:
        ax.axis("off"); continue
    y = np.arange(len(sub))
    ax.axvline(1, color="grey", lw=1, ls="--")
    for i, row in sub.reset_index(drop=True).iterrows():
        ax.plot(row["HR"], i, "o", color=color, ms=8, zorder=3)
        if row["CI_lo"] is not None and not pd.isna(row["CI_lo"]):
            ax.plot([row["CI_lo"], row["CI_hi"]], [i, i], color=color, lw=2)
        ax.text(row["HR"], i + 0.15, f'{row["HR"]:.2f}', ha="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["covariate"].values, fontsize=9)
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.invert_yaxis()

fig.suptitle(f"HR Forest Plot — {best_model['label']}", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR + "04_ecoli_lev_hr_forest.png", dpi=150)
plt.show()
print("Saved: 04_ecoli_lev_hr_forest.png")
"""))

# ── Final summary ──────────────────────────────────────────────────────────────
cells.append(md("## Summary of Outputs"))
cells.append(code("""\
import os
outputs = [
    "04_ecoli_lev_msm_data.csv",
    "04_ecoli_lev_aic_table.csv",
    "04_ecoli_lev_hr_table.csv",
    "04_ecoli_lev_sojourn_profiles.csv",
    "04_ecoli_lev_state_occupancy.png",
    "04_ecoli_lev_diag_prevalence.png",
    "04_ecoli_lev_diag_km_vpc.png",
    "04_ecoli_lev_diag_obs_exp.csv",
    "04_ecoli_lev_diag_obs_exp.png",
    "04_ecoli_lev_hr_forest.png",
]
print("Output files:")
for f in outputs:
    path = OUT_DIR + f
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path):,} bytes" if exists else "NOT FOUND"
    print(f"  {'OK' if exists else '!!'} {f}  ({size})")

print(f"\\nBest model: {best_model['label']}")
print(aic_df[['model','AIC','dAIC']].to_string(index=False))
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {OUT_NB}")
print(f"Total cells: {len(cells)}")
