# """Nexlyra Project 4 — Day 5: Statistical Analysis & Logic Traps.

Runs the descriptive statistics, OLS/correlation, Kolmogorov-Smirnov,
and Isolation Forest checks against the cleaned telemetry data, and
documents why several of these tests are statistically undefined or
non-informative once the relevant telemetry fields turn out to have
zero variance (a "logic trap" built into the Day 5 exercise). Also
computes the deterministic damages figure and writes the final
board-review outputs.

"""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import gaussian_kde, ks_2samp, kurtosis, pearsonr, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_CSV = BASE_DIR / "nexlyra_telemetry_master.csv"
EXECUTIVE_CSV = BASE_DIR / "executive_summary.csv"
PHYSICS_CSV = BASE_DIR / "physics_violations.csv"

BOARD_CSV = BASE_DIR / "Board_Review_Dataset.csv"

FINAL_PARQUET = BASE_DIR / "Carol_Day5_Final_Parquet.parquet"

HTML_OUTPUT = BASE_DIR / "Carol_Day5_Low_Fuel_Styled.html"

FUEL_KDE_PNG = BASE_DIR / "Day5_Fuel_KDE.png"

BILLING_KDE_PNG = BASE_DIR / "Day5_Log_Billing_KDE.png"

CARTEL_FUEL_PNG = BASE_DIR / "Day5_Cartel_Expected_vs_Actual_Fuel.png"

RANDOM_STATE = 42

# ================================================================
# REFERENCE VALUES
# ================================================================

# Day 1 lookup values
EQUIPMENT_IDLE_FUEL = {"Heavy_Excavator": 2.0, "Concrete_Pump": 2.0, "Tower_Crane": 2.0}

# Day 4 Ghost Equipment rule
FRAUD_VENDOR = "Phantom_Leasing"
FRAUD_STATIC = 1
FRAUD_HOURS = 20.0

# Day 5 deterministic logic-trap rule
RULE_FUEL = 0.5
RULE_RPM = 2000.0


# ================================================================
# HELPER FUNCTIONS
# ================================================================


def section(task_no, title):
    """Print a formatted task-header banner for console output."""

    print("\n" + "=" * 75)
    print(f"TASK {task_no:02d} — {title}")
    print("=" * 75)


def check_file(path):
    """Raise FileNotFoundError with a clear message if path is missing."""

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"Place the file in the same folder as this script."
        )


def numeric(series):
    """Coerce a Series to numeric, turning unparseable values into NaN."""

    return pd.to_numeric(series, errors="coerce")


def save_plot(path):
    """Tighten layout, save the current matplotlib figure, and close it."""

    plt.tight_layout()

    plt.savefig(path, dpi=180, bbox_inches="tight")

    plt.close()

    print(f"Plot exported: {path.name}")


# ================================================================
# TASK 01
# ================================================================

section(1, "Statistical Analysis & Logic Traps")

print("Day 5 performs statistical validation and fraud-rule analysis.")

print("\nMethods used:")

print("1. Descriptive statistics")

print("2. Z-score")

print("3. KDE distribution analysis")

print("4. Pearson correlation")

print("5. OLS regression")

print("6. IQR outlier detection")

print("7. Skewness and kurtosis")

print("8. Log transformation")

print("9. Kolmogorov-Smirnov test")

print("10. Conditional probability")

print("11. Bayes' theorem")

print("12. Confusion matrix")

print("13. Isolation Forest")

print("14. Financial damage calculation")


# ================================================================
# TASK 02
# ================================================================

section(2, "Load Final SQL Query Exports into pandas")

check_file(MASTER_CSV)
check_file(EXECUTIVE_CSV)
check_file(PHYSICS_CSV)

print(f"Loading: {MASTER_CSV.name}")

df = pd.read_csv(MASTER_CSV, low_memory=False)

executive_df = pd.read_csv(EXECUTIVE_CSV)

physics_df = pd.read_csv(PHYSICS_CSV)

print(f"Master dataset shape: {df.shape}")

print(f"Executive summary shape: {executive_df.shape}")

print(f"Physics violations shape: {physics_df.shape}")


# ------------------------------------------------
# Convert numeric columns
# ------------------------------------------------

numeric_columns = [
    "Billed_Engine_Hours",
    "Billed_Amount_USD",
    "Engine_RPM",
    "Fuel_LPH",
    "Is_Static_GPS",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = numeric(df[column])


# ------------------------------------------------
# Convert date
# ------------------------------------------------

if "Telemetry_Date" in df.columns:

    df["Telemetry_Date"] = pd.to_datetime(df["Telemetry_Date"], errors="coerce")


# ------------------------------------------------
# Create Day 4 reference fraud label
# ------------------------------------------------

df["Reference_Fraud"] = (
    (df["Vendor"] == FRAUD_VENDOR)
    & (df["Is_Static_GPS"] == FRAUD_STATIC)
    & (df["Billed_Engine_Hours"] > FRAUD_HOURS)
).astype(np.int8)


print("\nReference Ghost Equipment rows:")

print(f"{df['Reference_Fraud'].sum():,}")


# ================================================================
# TASK 03
# ================================================================

section(3, "Mathematical Proof / Physics Trap Validation")

tower = df[df["Equipment_Type"].astype(str).str.strip().eq("Tower_Crane")].copy()

print(f"Tower_Crane rows: {len(tower):,}")

print(f"RPM mean: {tower['Engine_RPM'].mean():.8f}")

print(f"Fuel mean: {tower['Fuel_LPH'].mean():.8f}")

print(f"RPM variance: " f"{tower['Engine_RPM'].var(ddof=1):.12f}")

print(f"Fuel variance: " f"{tower['Fuel_LPH'].var(ddof=1):.12f}")

print(
    "\nThe statistical calculations below use the actual "
    "observed telemetry rather than assuming a desired result."
)


# ================================================================
# TASK 04
# ================================================================

section(4, "Global Mean Fuel_LPH for Tower_Cranes")

tower_mean_fuel = tower["Fuel_LPH"].mean()

print(f"Tower_Crane Mean Fuel_LPH: " f"{tower_mean_fuel:.8f}")


# ================================================================
# TASK 05
# ================================================================

section(5, "Global Variance and Standard Deviation")

tower_variance = tower["Fuel_LPH"].var(ddof=1)

tower_std = tower["Fuel_LPH"].std(ddof=1)

print(f"Variance: " f"{tower_variance:.12f}")

print(f"Standard Deviation: " f"{tower_std:.12f}")


# ================================================================
# TASK 06
# ================================================================

section(6, "Phantom_Leasing Fuel_LPH Z-Score")

phantom = df[df["Vendor"] == FRAUD_VENDOR].copy()

phantom_mean = phantom["Fuel_LPH"].mean()

print(f"Phantom_Leasing Mean Fuel_LPH: " f"{phantom_mean:.8f}")

if pd.isna(tower_std) or tower_std == 0:

    print("\nZ-Score is undefined because " "Tower_Crane Fuel_LPH has zero variance.")

else:

    phantom_z = (phantom_mean - tower_mean_fuel) / tower_std

    print(f"Manual Z-Score: " f"{phantom_z:.8f}")


# ================================================================
# TASK 07
# ================================================================

section(7, "Bell Curve KDE of Fuel_LPH")

fuel = df["Fuel_LPH"].dropna().astype(float)

plt.figure(figsize=(10, 6))

if fuel.nunique() > 1:

    sample = fuel.sample(min(len(fuel), 100000), random_state=RANDOM_STATE)

    x = np.linspace(sample.min(), sample.max(), 500)

    kde = gaussian_kde(sample)

    plt.plot(x, kde(x), linewidth=2)

else:

    plt.axvline(fuel.iloc[0], linewidth=2)

plt.xlabel("Fuel Consumption (LPH)")

plt.ylabel("Density")

plt.title("Fuel_LPH Distribution — KDE")

plt.grid(True, alpha=0.25)

save_plot(FUEL_KDE_PNG)


# ================================================================
# TASK 08
# ================================================================

section(8, "Overlay Zero-Fuel Reference Line")

plt.figure(figsize=(10, 6))

if fuel.nunique() > 1:

    sample = fuel.sample(min(len(fuel), 100000), random_state=RANDOM_STATE)

    x = np.linspace(sample.min(), sample.max(), 500)

    kde = gaussian_kde(sample)

    plt.plot(x, kde(x), linewidth=2)

else:

    plt.axvline(fuel.iloc[0], linewidth=2)


plt.axvline(0, linestyle="--", linewidth=2, label="Zero Fuel Reference")

plt.xlabel("Fuel Consumption (LPH)")

plt.ylabel("Density")

plt.title("Fuel_LPH Distribution with Zero-Fuel Reference")

plt.legend()

plt.grid(True, alpha=0.25)

save_plot(FUEL_KDE_PNG)


# ================================================================
# TASK 09
# ================================================================

section(9, "Pearson Correlation — Engine_RPM vs Fuel_LPH")

corr_data = df[["Engine_RPM", "Fuel_LPH"]].dropna()

if (
    len(corr_data) < 2
    or corr_data["Engine_RPM"].nunique() < 2
    or corr_data["Fuel_LPH"].nunique() < 2
):

    global_corr = np.nan

    print(
        "Pearson correlation is undefined "
        "because one or both variables have "
        "zero variance."
    )

else:

    global_corr, global_p = pearsonr(corr_data["Engine_RPM"], corr_data["Fuel_LPH"])

    print(f"Pearson r: " f"{global_corr:.8f}")

    print(f"p-value: " f"{global_p:.8g}")


# ================================================================
# TASK 10
# ================================================================

section(10, "Pearson Correlation for Fraud Subset")

fraud_data = df[df["Reference_Fraud"] == 1][["Engine_RPM", "Fuel_LPH"]].dropna()

print(f"Fraud subset rows: " f"{len(fraud_data):,}")

if (
    len(fraud_data) < 2
    or fraud_data["Engine_RPM"].nunique() < 2
    or fraud_data["Fuel_LPH"].nunique() < 2
):

    fraud_corr = np.nan

    print(
        "Fraud-subset Pearson correlation "
        "is undefined because one or both "
        "variables have zero variance."
    )

else:

    fraud_corr, fraud_p = pearsonr(fraud_data["Engine_RPM"], fraud_data["Fuel_LPH"])

    print(f"Fraud subset Pearson r: " f"{fraud_corr:.8f}")

    print(f"Fraud subset p-value: " f"{fraud_p:.8g}")


# ================================================================
# TASK 11
# ================================================================

section(11, "Prepare statsmodels OLS Regression")

print("Model:")

print("Fuel_LPH = Intercept + " "Slope × Engine_RPM")

print(
    "\nThe slope measures the estimated "
    "change in fuel consumption for "
    "one unit of RPM."
)


# ================================================================
# TASK 12
# ================================================================

section(12, "Run OLS Regression")

# Because the relevant telemetry variables (Engine_RPM, Fuel_LPH) had
# effectively zero variance for this equipment subset, conventional
# OLS and correlation statistics are undefined/not informative here.
# We only fit the model when there is enough variation to do so, and
# otherwise report that explicitly rather than forcing a result.
ols_data = df[["Engine_RPM", "Fuel_LPH"]].dropna()

global_ols = None

if len(ols_data) >= 2 and ols_data["Engine_RPM"].nunique() >= 2:

    X = sm.add_constant(ols_data["Engine_RPM"])

    y = ols_data["Fuel_LPH"]

    global_ols = sm.OLS(y, X).fit()

    print(global_ols.summary())

else:

    print(
        "Global OLS slope cannot be uniquely "
        "estimated because Engine_RPM has "
        "zero variance."
    )


# ------------------------------------------------
# Fraud OLS
# ------------------------------------------------

fraud_ols = None

if len(fraud_data) >= 2 and fraud_data["Engine_RPM"].nunique() >= 2:

    X_fraud = sm.add_constant(fraud_data["Engine_RPM"])

    y_fraud = fraud_data["Fuel_LPH"]

    fraud_ols = sm.OLS(y_fraud, X_fraud).fit()

    print("\nFraud subset OLS:")

    print(fraud_ols.summary())

else:

    print(
        "\nFraud-subset OLS slope cannot "
        "be uniquely estimated because "
        "Engine_RPM has zero variance."
    )


# ================================================================
# TASK 13
# ================================================================

section(13, "Extract OLS Slope")

if global_ols is None:

    print("Global OLS slope: UNDEFINED")

else:

    global_slope = global_ols.params["Engine_RPM"]

    print(f"Global OLS slope: " f"{global_slope:.12f}")


if fraud_ols is None:

    print("Fraud-subset OLS slope: UNDEFINED")

else:

    fraud_slope = fraud_ols.params["Engine_RPM"]

    print(f"Fraud-subset OLS slope: " f"{fraud_slope:.12f}")


# ================================================================
# TASK 14
# ================================================================

section(14, "R-Squared")

if global_ols is None:

    print("Global R-squared: UNDEFINED")

else:

    print(f"Global R-squared: " f"{global_ols.rsquared:.8f}")


if fraud_ols is None:

    print("Fraud-subset R-squared: UNDEFINED")

else:

    print(f"Fraud-subset R-squared: " f"{fraud_ols.rsquared:.8f}")


# ================================================================
# TASK 15
# ================================================================

section(15, "IQR Outlier Detection on Engine_RPM")

rpm = df["Engine_RPM"].dropna()

q1 = rpm.quantile(0.25)

q3 = rpm.quantile(0.75)

iqr = q3 - q1

upper_bound = q3 + (1.5 * iqr)

df["RPM_Outlier"] = (df["Engine_RPM"] > upper_bound).astype(np.int8)

print(f"RPM outliers identified: " f"{df['RPM_Outlier'].sum():,}")


# ================================================================
# TASK 16
# ================================================================

section(16, "Calculate Q1 and Q3")

print(f"Q1: {q1:.8f}")

print(f"Q3: {q3:.8f}")


# ================================================================
# TASK 17
# ================================================================

section(17, "Calculate IQR")

print("IQR = Q3 - Q1")

print(f"IQR: {iqr:.8f}")


# ================================================================
# TASK 18
# ================================================================

section(18, "Extreme Upper Bound")

print("Formula:")

print("Q3 + (1.5 × IQR)")

print(f"Extreme upper bound: " f"{upper_bound:.8f}")


# ================================================================
# TASK 19
# ================================================================

section(19, "Flag Over-Rev Outliers")

df["Over_Rev_Outlier"] = (df["Engine_RPM"] > upper_bound).astype(np.int8)

overrev_count = int(df["Over_Rev_Outlier"].sum())

print(f"Over-Rev Outliers: " f"{overrev_count:,}")


# ================================================================
# TASK 20
# ================================================================

section(20, "Percentage of Over-Rev Outliers Belonging to Phantom_Leasing")

if overrev_count == 0:

    print("No Over-Rev Outliers were identified.")

else:

    phantom_overrev = int(
        ((df["Over_Rev_Outlier"] == 1) & (df["Vendor"] == FRAUD_VENDOR)).sum()
    )

    percentage = phantom_overrev / overrev_count * 100

    print(f"Phantom Over-Rev outliers: " f"{phantom_overrev:,}")

    print(f"Phantom percentage: " f"{percentage:.2f}%")


# ================================================================
# TASK 21
# ================================================================

section(21, "numpy.histogram Fuel_LPH Frequency Bins")

fuel_values = df["Fuel_LPH"].dropna().to_numpy()

counts, edges = np.histogram(fuel_values, bins=10)

hist_table = pd.DataFrame(
    {"Bin_Left": edges[:-1], "Bin_Right": edges[1:], "Frequency": counts}
)

print(hist_table.to_string(index=False))


# ================================================================
# TASK 22
# ================================================================

section(22, "Fuel_LPH Skewness")

if len(fuel_values) >= 3:

    fuel_skew = skew(fuel_values, bias=False)

    print(f"Skewness: " f"{fuel_skew:.8f}")

else:

    fuel_skew = np.nan

    print("Skewness cannot be calculated.")


# ================================================================
# TASK 23
# ================================================================

section(23, "Fuel_LPH Kurtosis")

if len(fuel_values) >= 4:

    fuel_kurtosis = kurtosis(fuel_values, fisher=True, bias=False)

    print(f"Excess Kurtosis: " f"{fuel_kurtosis:.8f}")

else:

    fuel_kurtosis = np.nan

    print("Kurtosis cannot be calculated.")


# ================================================================
# TASK 24
# ================================================================

section(24, "Interpret Skewness and Normalization")

print("High positive skewness means that the " "distribution has a longer right tail.")

print(
    "In billing data, this can happen when "
    "a small number of invoices are much "
    "larger than typical invoices."
)

print(
    "A logarithmic transformation can reduce "
    "the influence of extreme positive values."
)


# ================================================================
# TASK 25
# ================================================================

section(25, "Logarithmic Transformation of Billing")

df["Log_Billed_Amount_USD"] = np.log1p(df["Billed_Amount_USD"].clip(lower=0))

print(
    df[["Billed_Amount_USD", "Log_Billed_Amount_USD"]].head(10).to_string(index=False)
)


# ================================================================
# TASK 26
# ================================================================

section(26, "Billing Bell Curve After Log Transformation")

log_billing = df["Log_Billed_Amount_USD"].dropna()

plt.figure(figsize=(10, 6))

if log_billing.nunique() > 1:

    sample = log_billing.sample(
        min(len(log_billing), 100000), random_state=RANDOM_STATE
    )

    x = np.linspace(sample.min(), sample.max(), 500)

    kde = gaussian_kde(sample)

    plt.plot(x, kde(x), linewidth=2)

else:

    plt.axvline(log_billing.iloc[0], linewidth=2)

plt.xlabel("log1p(Billed Amount USD)")

plt.ylabel("Density")

plt.title("Log-Transformed Billing Distribution")

plt.grid(True, alpha=0.25)

save_plot(BILLING_KDE_PNG)


# ================================================================
# TASK 27
# ================================================================

section(27, "KS-Test — Moving vs Static Fuel")

moving_fuel = df.loc[df["Is_Static_GPS"] == 0, "Fuel_LPH"].dropna()

static_fuel = df.loc[df["Is_Static_GPS"] == 1, "Fuel_LPH"].dropna()

print(f"Moving observations: " f"{len(moving_fuel):,}")

print(f"Static observations: " f"{len(static_fuel):,}")

if len(moving_fuel) == 0 or len(static_fuel) == 0:

    ks_stat = np.nan
    ks_p = np.nan

    print("KS-Test cannot be performed.")

else:

    ks_result = ks_2samp(moving_fuel, static_fuel)

    ks_stat = ks_result.statistic

    ks_p = ks_result.pvalue


# ================================================================
# TASK 28
# ================================================================

section(28, "Document KS-Test Statistic and p-value")

if pd.isna(ks_stat):

    print("KS statistic: undefined")

    print("KS p-value: undefined")

else:

    print(f"KS statistic: " f"{ks_stat:.8f}")

    print(f"KS p-value: " f"{ks_p:.8g}")

    if ks_p < 0.05:

        print("\nAt the 5% level, reject the " "same-distribution null hypothesis.")

    else:

        print(
            "\nAt the 5% level, do not reject " "the same-distribution null hypothesis."
        )


# ================================================================
# TASK 29
# ================================================================

section(29, "P(Fraud | Fuel < 0.5 LPH)")

low_fuel = df["Fuel_LPH"] < RULE_FUEL

low_fuel_count = int(low_fuel.sum())

fraud_low_fuel = int(((df["Reference_Fraud"] == 1) & low_fuel).sum())

if low_fuel_count == 0:

    p_fraud_given_low = np.nan

    print("No Fuel_LPH < 0.5 observations.")

else:

    p_fraud_given_low = fraud_low_fuel / low_fuel_count

    print(f"P(Fraud | Fuel < 0.5): " f"{p_fraud_given_low:.8f}")


# ================================================================
# TASK 30
# ================================================================

section(30, "P(Fuel < 0.5 LPH | Fraud)")

fraud_count = int(df["Reference_Fraud"].sum())

if fraud_count == 0:

    p_low_given_fraud = np.nan

    print("No reference-fraud records.")

else:

    p_low_given_fraud = fraud_low_fuel / fraud_count

    print(f"P(Fuel < 0.5 | Fraud): " f"{p_low_given_fraud:.8f}")


# ================================================================
# TASK 31
# ================================================================

section(31, "Bayes Theorem — Low Fuel + High RPM")

evidence = (df["Fuel_LPH"] < RULE_FUEL) & (df["Engine_RPM"] > RULE_RPM)

evidence_count = int(evidence.sum())

fraud_and_evidence = int(((df["Reference_Fraud"] == 1) & evidence).sum())

if evidence_count == 0:

    bayes_probability = np.nan

    print("No records satisfy:")

    print("Fuel < 0.5 AND RPM > 2000")

    print(
        "\nObserved posterior probability "
        "cannot be estimated because the "
        "evidence event has zero observations."
    )

else:

    bayes_probability = fraud_and_evidence / evidence_count

    print(f"P(Fraud | low fuel + high RPM): " f"{bayes_probability:.8f}")

print("\nBayes formula:")

print("P(Fraud | Evidence) = " "P(Evidence | Fraud) × P(Fraud) " "/ P(Evidence)")


# ================================================================
# TASK 32
# ================================================================

section(32, "Confusion Matrix — Deterministic Rule")

df["Deterministic_Prediction"] = (
    (df["Fuel_LPH"] < RULE_FUEL) & (df["Engine_RPM"] > RULE_RPM)
).astype(np.int8)

cm = confusion_matrix(
    df["Reference_Fraud"], df["Deterministic_Prediction"], labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")

print(cm)

print(f"\nTrue Negative: " f"{tn:,}")

print(f"False Positive: " f"{fp:,}")

print(f"False Negative: " f"{fn:,}")

print(f"True Positive: " f"{tp:,}")


# ================================================================
# TASK 33
# ================================================================

section(33, "Precision")

precision = precision_score(
    df["Reference_Fraud"], df["Deterministic_Prediction"], zero_division=0
)

print(f"Precision: " f"{precision:.8f}")


# ================================================================
# TASK 34
# ================================================================

section(34, "Recall")

recall = recall_score(
    df["Reference_Fraud"], df["Deterministic_Prediction"], zero_division=0
)

print(f"Recall: " f"{recall:.8f}")


# ================================================================
# TASK 35
# ================================================================

section(35, "F1-Score")

f1 = f1_score(df["Reference_Fraud"], df["Deterministic_Prediction"], zero_division=0)

print(f"F1-Score: " f"{f1:.8f}")


# ================================================================
# TASK 36
# ================================================================

section(36, "False Positive Analysis")

false_positive_df = df[
    (df["Reference_Fraud"] == 0) & (df["Deterministic_Prediction"] == 1)
].copy()

print(f"False positives: " f"{len(false_positive_df):,}")

if len(false_positive_df) == 0:

    print("No legitimate records triggered " "the deterministic rule.")

else:

    print("\nPotential false-positive records:")

    print(
        false_positive_df[
            ["Log_ID", "Vendor", "Equipment_Type", "Engine_RPM", "Fuel_LPH"]
        ]
        .head(20)
        .to_string(index=False)
    )


# ================================================================
# TASK 37
# ================================================================

section(37, "False Negative Analysis")

false_negative_df = df[
    (df["Reference_Fraud"] == 1) & (df["Deterministic_Prediction"] == 0)
].copy()

print(f"False negatives: " f"{len(false_negative_df):,}")

if len(false_negative_df) == 0:

    print("No reference-fraud records escaped " "the deterministic rule.")

else:

    print("Potential false-negative records:")

    print(
        false_negative_df[
            ["Log_ID", "Vendor", "Equipment_Type", "Engine_RPM", "Fuel_LPH"]
        ]
        .head(20)
        .to_string(index=False)
    )


# ================================================================
# TASK 38
# ================================================================

section(38, "Isolation Forest — RPM + Fuel")

# Caveat: with RPM and Fuel_LPH essentially constant for this
# equipment subset, Isolation Forest has no real spread to model and
# tends to flag every row as an outlier. That result is a symptom of
# the zero-variance inputs, not meaningful evidence of fraud on its
# own — it should not be presented as such.
ml_data = df[["Engine_RPM", "Fuel_LPH"]].dropna().copy()

print(f"Rows used: " f"{len(ml_data):,}")

isolation_forest = IsolationForest(
    n_estimators=200, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1
)

ml_labels = isolation_forest.fit_predict(ml_data[["Engine_RPM", "Fuel_LPH"]])

ml_scores = isolation_forest.decision_function(ml_data[["Engine_RPM", "Fuel_LPH"]])

ml_data["IsolationForest_Label"] = ml_labels

ml_data["IsolationForest_Score"] = ml_scores

ml_data["ML_Outlier"] = (ml_data["IsolationForest_Label"] == -1).astype(np.int8)

print(f"Isolation Forest outliers: " f"{ml_data['ML_Outlier'].sum():,}")


# ------------------------------------------------
# Add results back to main dataframe
# ------------------------------------------------

df["ML_Outlier"] = 0

df["IsolationForest_Score"] = np.nan

df.loc[ml_data.index, "ML_Outlier"] = ml_data["ML_Outlier"]

df.loc[ml_data.index, "IsolationForest_Score"] = ml_data["IsolationForest_Score"]


# ================================================================
# TASK 39
# ================================================================

section(39, "Compare ML vs Deterministic Anomalies")

deterministic_count = int(df["Deterministic_Prediction"].sum())

ml_count = int(df["ML_Outlier"].sum())

intersection_count = int(
    ((df["ML_Outlier"] == 1) & (df["Deterministic_Prediction"] == 1)).sum()
)

print(f"Deterministic anomalies: " f"{deterministic_count:,}")

print(f"Isolation Forest outliers: " f"{ml_count:,}")

print(f"Intersection: " f"{intersection_count:,}")


# ================================================================
# TASK 40
# ================================================================

section(40, "Intersection Percentage")

if deterministic_count == 0:

    intersection_percentage = np.nan

    print(
        "No deterministic anomalies exist, " "so intersection percentage is undefined."
    )

else:

    intersection_percentage = intersection_count / deterministic_count * 100

    print(f"Intersection percentage: " f"{intersection_percentage:.2f}%")


# ================================================================
# TASK 41
# ================================================================

section(41, "Function to Calculate Exact USD Amount Stolen")


def calculate_damage(row):
    """

    Returns the billed amount when:

        Fuel_LPH < 0.5
        AND
        Engine_RPM > 2000

    Otherwise returns 0.

    """

    fuel_value = row["Fuel_LPH"]

    rpm_value = row["Engine_RPM"]

    billed_value = row["Billed_Amount_USD"]

    if (
        pd.notna(fuel_value)
        and pd.notna(rpm_value)
        and pd.notna(billed_value)
        and fuel_value < RULE_FUEL
        and rpm_value > RULE_RPM
    ):

        return float(billed_value)

    return 0.0


print("Damage calculation function created.")


# ================================================================
# TASK 42
# ================================================================

section(42, "Calculate Total_Damages_USD")

df["Total_Damages_USD"] = df.apply(calculate_damage, axis=1)

damage_rows = int((df["Total_Damages_USD"] > 0).sum())

print(f"Rows with damages > $0: " f"{damage_rows:,}")

if damage_rows > 0:

    print(
        df[
            [
                "Log_ID",
                "Vendor",
                "Engine_RPM",
                "Fuel_LPH",
                "Billed_Amount_USD",
                "Total_Damages_USD",
            ]
        ]
        .query("Total_Damages_USD > 0")
        .head(20)
        .to_string(index=False)
    )


# ================================================================
# TASK 43
# ================================================================

section(43, "Final Financial Impact")

total_damages = df["Total_Damages_USD"].sum()

print(f"Total_Damages_USD: " f"${total_damages:,.2f}")

print(
    "\nThis represents billing attached "
    "specifically to the Day 5 deterministic "
    "condition: Fuel < 0.5 LPH and RPM > 2000."
)


# ================================================================
# TASK 44
# ================================================================

section(44, "Expected Fuel vs Actual Fuel — Phantom_Leasing")

cartel = df[df["Vendor"] == FRAUD_VENDOR].copy()

cartel["Expected_Idle_Fuel_LPH"] = cartel["Equipment_Type"].map(EQUIPMENT_IDLE_FUEL)

cartel_summary = (
    cartel.groupby("Equipment_Type", dropna=False)
    .agg(
        Expected_Fuel_LPH=("Expected_Idle_Fuel_LPH", "mean"),
        Actual_Fuel_LPH=("Fuel_LPH", "mean"),
        Records=("Log_ID", "count"),
    )
    .reset_index()
)

print(cartel_summary.to_string(index=False))


# ------------------------------------------------
# Grouped bar chart
# ------------------------------------------------

if len(cartel_summary) > 0:

    x = np.arange(len(cartel_summary))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2, cartel_summary["Expected_Fuel_LPH"], width, label="Expected Fuel"
    )

    plt.bar(
        x + width / 2, cartel_summary["Actual_Fuel_LPH"], width, label="Actual Fuel"
    )

    plt.xticks(x, cartel_summary["Equipment_Type"], rotation=20)

    plt.xlabel("Equipment Type")

    plt.ylabel("Fuel LPH")

    plt.title("Phantom_Leasing — Expected vs Actual Fuel")

    plt.legend()

    plt.grid(True, axis="y", alpha=0.25)

    save_plot(CARTEL_FUEL_PNG)


# ================================================================
# TASK 45
# ================================================================

section(45, "Create Highest-Confidence Fraud Subset")

# This is the intersection of:
#
# 1. Day 4 Ghost Equipment reference rule
# 2. Day 5 deterministic physics rule
# 3. Isolation Forest anomaly
#
# It is intentionally conservative.

high_confidence = df[
    (df["Reference_Fraud"] == 1)
    & (df["Deterministic_Prediction"] == 1)
    & (df["ML_Outlier"] == 1)
].copy()

print(f"Highest-confidence rows: " f"{len(high_confidence):,}")


# ================================================================
# TASK 46
# ================================================================

section(46, "OLS Physics Proof — Non-Technical Explanation")

if fraud_ols is None:

    print(
        "The observed fraud subset has no variation "
        "in Engine_RPM, so a unique numerical OLS "
        "slope cannot be estimated."
    )

    print(
        "\nInstead, the data should be described "
        "as showing effectively constant RPM values "
        "within the observed fraud subset."
    )

else:

    print(
        "OLS measures whether fuel consumption changes "
        "systematically with engine RPM."
    )

    print(
        "A slope near zero would mean fuel consumption "
        "changes very little when RPM changes."
    )

    print(
        "The actual OLS coefficient above should be " "used as the statistical result."
    )


# ================================================================
# TASK 47
# ================================================================

section(47, "Export Board_Review_Dataset.csv")

high_confidence.to_csv(BOARD_CSV, index=False)

print(f"Exported: " f"{BOARD_CSV.name}")

print(f"Rows exported: " f"{len(high_confidence):,}")


# ================================================================
# TASK 48
# ================================================================

section(48, "pandas Styling — Low Fuel Output")

low_fuel_output = (
    df[
        [
            "Log_ID",
            "Vendor",
            "Equipment_Type",
            "Engine_RPM",
            "Fuel_LPH",
            "Billed_Amount_USD",
        ]
    ]
    .sort_values("Fuel_LPH", ascending=True)
    .head(100)
    .copy()
)

styled_low_fuel = low_fuel_output.style.highlight_min(
    subset=["Fuel_LPH"], axis=0
).format(
    {"Engine_RPM": "{:.2f}", "Fuel_LPH": "{:.4f}", "Billed_Amount_USD": "${:,.2f}"}
)

print("DataFrame styling applied using " "pandas Styler.highlight_min().")


# ================================================================
# TASK 49
# ================================================================

section(49, "Render Styled DataFrame to Static HTML")

styled_low_fuel.to_html(HTML_OUTPUT, index=False)

print(f"HTML exported: " f"{HTML_OUTPUT.name}")


# ================================================================
# TASK 50
# ================================================================

section(50, "Clear Variables From Memory")

# Preserve final dataframe
# for Task 51.

final_df = df.copy()

# Remove temporary objects.

del df
del tower
del phantom
del corr_data
del fraud_data
del ols_data
del ml_data
del cartel
del low_fuel_output
del styled_low_fuel
del executive_df
del physics_df
del false_positive_df
del false_negative_df
del high_confidence

gc.collect()

print("Temporary analysis variables cleared.")

print("Final dataframe retained as final_df.")


# ================================================================
# TASK 51
# ================================================================

section(51, "Prepare Final Parquet for Power BI")

final_df.to_parquet(FINAL_PARQUET, index=False, engine="pyarrow")

print(f"Final Parquet created: " f"{FINAL_PARQUET.name}")

print(f"Final dataframe shape: " f"{final_df.shape}")


# ------------------------------------------------
# Validate Parquet
# ------------------------------------------------

parquet_check = pd.read_parquet(FINAL_PARQUET, engine="pyarrow")

print(f"Parquet rows: " f"{len(parquet_check):,}")

print(f"Parquet columns: " f"{len(parquet_check.columns):,}")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)

print("DAY 5 — FINAL SUMMARY")

print("=" * 75)

print(f"Master rows analyzed: " f"{len(final_df):,}")

print(f"Reference Ghost Equipment rows: " f"{final_df['Reference_Fraud'].sum():,}")

print(
    f"Deterministic logic-trap rows: " f"{final_df['Deterministic_Prediction'].sum():,}"
)

print(f"Isolation Forest outliers: " f"{final_df['ML_Outlier'].sum():,}")

final_intersection = (
    (final_df["Reference_Fraud"] == 1)
    & (final_df["Deterministic_Prediction"] == 1)
    & (final_df["ML_Outlier"] == 1)
).sum()

print(f"Highest-confidence intersection: " f"{final_intersection:,}")

print(f"Total_Damages_USD: " f"${final_df['Total_Damages_USD'].sum():,.2f}")

print(f"Precision: " f"{precision:.8f}")

print(f"Recall: " f"{recall:.8f}")

print(f"F1-Score: " f"{f1:.8f}")

if not pd.isna(ks_stat):

    print(f"KS statistic: " f"{ks_stat:.8f}")

    print(f"KS p-value: " f"{ks_p:.8g}")

print("\nOutput files:")

for file in [
    BOARD_CSV,
    HTML_OUTPUT,
    FUEL_KDE_PNG,
    BILLING_KDE_PNG,
    CARTEL_FUEL_PNG,
    FINAL_PARQUET,
]:

    print(f"{file.name} → " f"{file.exists()}")


print("\n" + "=" * 75)

print("DAY 5 PROCESSING FINISHED")

print("=" * 75)
