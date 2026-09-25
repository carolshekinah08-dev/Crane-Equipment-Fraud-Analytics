# 
# """Nexlyra Project 4 — Day 2: Python Data Engineering & Logic Trap Extraction.

Loads the raw 1,000,000-row crane telemetry export, benchmarks an
unoptimized load against a memory-optimized load, removes duplicate
log entries, repairs malformed embedded JSON payloads, and writes a
cleaned baseline dataset to Parquet for use in later project days.

"""

import gc
import json
import re
import time

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "nexlyra_crane_fraud_1M_carol.csv"

PARQUET_FILE = "Carol_Crane_Clean_Baseline.parquet"

print("=" * 70)
print("NEXLYRA PROJECT 4 — DAY 2")
print("Python Data Engineering & Logic Trap Extraction")
print("=" * 70)


# ============================================================
# TASK 01 — Import pandas, numpy, json, re
# ============================================================

print("\nTASK 01 — Libraries imported")
print("pandas version:", pd.__version__)


# ============================================================
# TASK 02 — Attempt to load the entire 1M CSV WITHOUT
# optimization and observe memory consumption
# ============================================================

print("\n" + "=" * 70)
print("TASK 02 — Unoptimized 1M-row load")
print("=" * 70)

start_time = time.time()

df_raw = pd.read_csv(CSV_FILE)

load_time = time.time() - start_time

print("\nUnoptimized dataset loaded.")
print("Shape:", df_raw.shape)

print("\nUnoptimized memory usage:")
df_raw.info(memory_usage="deep")

unoptimized_memory_mb = df_raw.memory_usage(deep=True).sum() / 1024**2

print(f"\nTOTAL UNOPTIMIZED MEMORY: " f"{unoptimized_memory_mb:.2f} MB")

print(f"Load time: {load_time:.2f} seconds")


# ============================================================
# TASK 03 — Observe RAM consumption
# Already displayed above using df.info(memory_usage='deep')
# ============================================================


# ============================================================
# TASK 04 — Optimized dtype mapping
#
# The task asks to consult the Nexlyra AI Code Model about
# mapping dtypes such as float32 and categorical.
#
# The optimized dtype strategy is implemented below.
# ============================================================

print("\n" + "=" * 70)
print("TASK 04 — Optimized dtype mapping")
print("=" * 70)

print("""
Optimization strategy:
- Billed_Engine_Hours -> float32
- Billed_Amount_USD -> cleaned later because source contains strings
- Site_Code -> category
- Vendor -> category
- Equipment_Type -> category
- IDs remain text
- Telemetry_Date remains text initially because formats are mixed
- IoT_Telemetry_JSON remains text initially
""")


# ============================================================
# TASK 05 — Implement optimized read_csv
# ============================================================

print("\n" + "=" * 70)
print("TASK 05 — Optimized CSV loading")
print("=" * 70)

# Delete the unoptimized dataframe before loading optimized version
del df_raw
gc.collect()

optimized_dtypes = {
    "Log_ID": "string",
    "Site_Code": "category",
    "Vendor": "category",
    "Equipment_Type": "category",
    "Billed_Engine_Hours": "float32",
    "Telemetry_Date": "string",
    "Billed_Amount_USD": "string",
    "IoT_Telemetry_JSON": "string",
}

start_time = time.time()

df = pd.read_csv(CSV_FILE, dtype=optimized_dtypes)

load_time = time.time() - start_time

print("Optimized dataset loaded.")
print("Shape:", df.shape)
print(f"Load time: {load_time:.2f} seconds")


# ============================================================
# TASK 06 — Verify memory footprint
# ============================================================

print("\n" + "=" * 70)
print("TASK 06 — Optimized memory footprint")
print("=" * 70)

df.info(memory_usage="deep")

optimized_memory_mb = df.memory_usage(deep=True).sum() / 1024**2

print(f"\nTOTAL OPTIMIZED MEMORY: " f"{optimized_memory_mb:.2f} MB")

if unoptimized_memory_mb > 0:
    reduction = (
        (unoptimized_memory_mb - optimized_memory_mb) / unoptimized_memory_mb
    ) * 100

    print(f"MEMORY REDUCTION: {reduction:.2f}%")


# ============================================================
# TASK 07 — Identify mixed datetime formats
# ============================================================

print("\n" + "=" * 70)
print("TASK 07 — Identify mixed Telemetry_Date formats")
print("=" * 70)

date_text = df["Telemetry_Date"].astype("string")

us_date_mask = date_text.str.contains(r"^\d{1,2}/\d{1,2}/\d{4}", regex=True, na=False)

iso_date_mask = date_text.str.contains(r"^\d{4}-\d{2}-\d{2}", regex=True, na=False)

print("US-style date rows:", us_date_mask.sum())
print("ISO-style date rows:", iso_date_mask.sum())

print("\nSample date values:")
print(df["Telemetry_Date"].head(10).to_string(index=False))


# ============================================================
# TASK 08 — Vectorized mixed datetime parsing
# ============================================================

print("\n" + "=" * 70)
print("TASK 08 — Mixed datetime parsing strategy")
print("=" * 70)

print("""
Using pandas vectorized pd.to_datetime with
format='mixed' so ISO and US-style dates
can be handled in one operation.
""")


# ============================================================
# TASK 09 — Convert Telemetry_Date to datetime64
# ============================================================

print("\n" + "=" * 70)
print("TASK 09 — Convert Telemetry_Date")
print("=" * 70)

df["Telemetry_Date"] = pd.to_datetime(
    df["Telemetry_Date"], format="mixed", errors="coerce"
)

print("Telemetry_Date dtype:")
print(df["Telemetry_Date"].dtype)

print("\nSample converted dates:")
print(df["Telemetry_Date"].head())


# ============================================================
# TASK 10 — Extract Hour_of_Day and Day_of_Week
# ============================================================

print("\n" + "=" * 70)
print("TASK 10 — Hour_of_Day and Day_of_Week")
print("=" * 70)

df["Hour_of_Day"] = df["Telemetry_Date"].dt.hour.astype("float32")

df["Day_of_Week"] = df["Telemetry_Date"].dt.dayofweek.astype("float32")

print(df[["Telemetry_Date", "Hour_of_Day", "Day_of_Week"]].head())


# ============================================================
# TASK 11 — Check for NaNs after time parsing
# ============================================================

print("\n" + "=" * 70)
print("TASK 11 — Null check after datetime parsing")
print("=" * 70)

date_null_count = df["Telemetry_Date"].isnull().sum()

print("Telemetry_Date null values:", date_null_count)

print("\nFull null summary:")
print(df.isnull().sum())


# ============================================================
# TASK 12 — Identify non-numeric characters in billing
# ============================================================

print("\n" + "=" * 70)
print("TASK 12 — Billing non-numeric character detection")
print("=" * 70)

billing_text = df["Billed_Amount_USD"].astype("string")

non_numeric_mask = billing_text.str.contains(r"[^0-9.\-]", regex=True, na=False)

print("Rows containing non-numeric characters:", non_numeric_mask.sum())

print("\nExamples:")
print(df.loc[non_numeric_mask, "Billed_Amount_USD"].head(10))


# ============================================================
# TASK 13 — Strip USD and commas using regex
# ============================================================

print("\n" + "=" * 70)
print("TASK 13 — Clean Billed_Amount_USD")
print("=" * 70)

df["Billed_Amount_USD"] = (
    df["Billed_Amount_USD"]
    .astype("string")
    .str.replace(r"USD\s*", "", regex=True)
    .str.replace(",", "", regex=False)
    .str.strip()
)

print("\nCleaned billing examples:")
print(df["Billed_Amount_USD"].head(10))


# ============================================================
# TASK 14 — Cast billing to float32
# ============================================================

print("\n" + "=" * 70)
print("TASK 14 — Convert billing to float32")
print("=" * 70)

df["Billed_Amount_USD"] = pd.to_numeric(
    df["Billed_Amount_USD"], errors="coerce"
).astype("float32")

print("Billed_Amount_USD dtype:", df["Billed_Amount_USD"].dtype)


# ============================================================
# TASK 15 — Calculate total billing
# ============================================================

print("\n" + "=" * 70)
print("TASK 15 — Total Billed_Amount_USD")
print("=" * 70)

total_billing = df["Billed_Amount_USD"].sum()

print(f"TOTAL BILLED AMOUNT: ${total_billing:,.2f}")


# ============================================================
# TASK 16 — Search for negative or zero billing
# ============================================================

print("\n" + "=" * 70)
print("TASK 16 — Negative / zero billing check")
print("=" * 70)

negative_or_zero = df[df["Billed_Amount_USD"] <= 0]

print("Negative or zero billing records:", len(negative_or_zero))

if len(negative_or_zero) > 0:
    print(negative_or_zero.head())


# ============================================================
# TASK 17 — Custom duplicate function based on Log_ID
# ============================================================

print("\n" + "=" * 70)
print("TASK 17 — Duplicate detection function")
print("=" * 70)


def remove_duplicate_log_ids(dataframe):
    """
    Removes duplicate records based on Log_ID.
    Keeps the first occurrence.
    Returns cleaned dataframe and number removed.
    """

    before = len(dataframe)

    cleaned = dataframe.drop_duplicates(subset=["Log_ID"], keep="first").copy()

    removed = before - len(cleaned)

    return cleaned, removed


duplicate_count = df["Log_ID"].duplicated().sum()

print("Duplicate Log_ID records found:", duplicate_count)


# ============================================================
# TASK 18 — Apply duplicate drop function
# ============================================================

print("\n" + "=" * 70)
print("TASK 18 — Remove duplicate Log_ID records")
print("=" * 70)

df, removed_rows = remove_duplicate_log_ids(df)

print("Total duplicate rows removed:", removed_rows)

print("Current dataframe shape:", df.shape)


# ============================================================
# TASK 19 — Inspect malformed IoT JSON
# ============================================================

print("\n" + "=" * 70)
print("TASK 19 — Inspect malformed IoT JSON")
print("=" * 70)

json_text = df["IoT_Telemetry_JSON"].astype("string")

# Look specifically for missing comma between rpm_val
# and fuel_val.
broken_json_mask = json_text.str.contains(
    r'"rpm_val"\s*:\s*[-+]?\d+(?:\.\d+)?\s+"fuel_val"', regex=True, na=False
)

broken_json_count = broken_json_mask.sum()

print("Potential malformed JSON rows:", broken_json_count)

if broken_json_count > 0:
    print("\nExamples of malformed JSON:")
    print(df.loc[broken_json_mask, "IoT_Telemetry_JSON"].head(5).to_string(index=False))


# ============================================================
# TASK 20 — Regex repair function
# ============================================================

print("\n" + "=" * 70)
print("TASK 20 — JSON repair regex")
print("=" * 70)


def repair_json_string(value):
    """
    Repairs a known malformed pattern where the comma
    between rpm_val and fuel_val is missing.
    """

    if pd.isna(value):
        return value

    text = str(value)

    repaired = re.sub(
        r'("rpm_val"\s*:\s*[-+]?\d+(?:\.\d+)?)\s+("fuel_val"\s*:)', r"\1, \2", text
    )

    return repaired


print("JSON repair function created.")


# ============================================================
# TASK 21 — Apply JSON repair to entire column
# ============================================================

print("\n" + "=" * 70)
print("TASK 21 — Apply JSON repair")
print("=" * 70)

df["IoT_Telemetry_JSON"] = (
    df["IoT_Telemetry_JSON"].apply(repair_json_string).astype("string")
)

print("JSON repair applied to entire column.")


# ============================================================
# TASK 22 — Safe json.loads() function
# ============================================================

print("\n" + "=" * 70)
print("TASK 22 — Safe JSON parsing function")
print("=" * 70)


def safe_json_load(value):
    """
    Safely parses JSON.
    Returns an empty dictionary if parsing fails.
    """

    if pd.isna(value):
        return {}

    try:
        return json.loads(str(value))

    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


print("Safe JSON parser created.")


# ============================================================
# TASK 23 — Apply safe parser
# ============================================================

print("\n" + "=" * 70)
print("TASK 23 — Parse JSON into dictionaries")
print("=" * 70)

df["Parsed_IoT"] = df["IoT_Telemetry_JSON"].apply(safe_json_load)

print("Parsed_IoT sample:")
print(df["Parsed_IoT"].head())


# ============================================================
# TASK 24 — Extract rpm_val -> Engine_RPM
# ============================================================

print("\n" + "=" * 70)
print("TASK 24 — Engine_RPM")
print("=" * 70)

df["Engine_RPM"] = pd.to_numeric(
    df["Parsed_IoT"].apply(
        lambda x: x.get("rpm_val", np.nan) if isinstance(x, dict) else np.nan
    ),
    errors="coerce",
).astype("float32")

print(df["Engine_RPM"].head())


# ============================================================
# TASK 25 — Extract fuel_val -> Fuel_LPH
# ============================================================

print("\n" + "=" * 70)
print("TASK 25 — Fuel_LPH")
print("=" * 70)

df["Fuel_LPH"] = pd.to_numeric(
    df["Parsed_IoT"].apply(
        lambda x: x.get("fuel_val", np.nan) if isinstance(x, dict) else np.nan
    ),
    errors="coerce",
).astype("float32")

print(df["Fuel_LPH"].head())


# ============================================================
# TASK 26 — Extract geo_b64 -> Encoded_GPS
# ============================================================

print("\n" + "=" * 70)
print("TASK 26 — Encoded_GPS")
print("=" * 70)

df["Encoded_GPS"] = (
    df["Parsed_IoT"]
    .apply(lambda x: x.get("geo_b64", np.nan) if isinstance(x, dict) else np.nan)
    .astype("string")
)

print(df["Encoded_GPS"].head())


# ============================================================
# TASK 27 — Identify JSON parsing failures
# ============================================================

print("\n" + "=" * 70)
print("TASK 27 — JSON parsing failures")
print("=" * 70)

json_parse_failed = (
    df["Engine_RPM"].isna() | df["Fuel_LPH"].isna() | df["Encoded_GPS"].isna()
)

json_failure_count = json_parse_failed.sum()

print("Rows with missing extracted telemetry:", json_failure_count)


# ============================================================
# TASK 28 — Fill missing Engine_RPM using median by
# Equipment_Type
# ============================================================

print("\n" + "=" * 70)
print("TASK 28 — Fill missing Engine_RPM")
print("=" * 70)

equipment_rpm_median = df.groupby("Equipment_Type", observed=True)[
    "Engine_RPM"
].transform("median")

missing_before = df["Engine_RPM"].isna().sum()

df["Engine_RPM"] = df["Engine_RPM"].fillna(equipment_rpm_median)

missing_after = df["Engine_RPM"].isna().sum()

print("Missing Engine_RPM before:", missing_before)

print("Missing Engine_RPM after:", missing_after)


# ============================================================
# TASK 29 — Drop heavy original JSON column
# ============================================================

print("\n" + "=" * 70)
print("TASK 29 — Drop original IoT JSON")
print("=" * 70)

df.drop(columns=["IoT_Telemetry_JSON"], inplace=True)

# Parsed dictionaries are also heavy and no longer needed
# after extracting the required fields.
df.drop(columns=["Parsed_IoT"], inplace=True)

gc.collect()

print("Original IoT_Telemetry_JSON column dropped.")


# ============================================================
# TASK 30 — Verify memory footprint
# ============================================================

print("\n" + "=" * 70)
print("TASK 30 — New memory footprint")
print("=" * 70)

df.info(memory_usage="deep")

new_memory_mb = df.memory_usage(deep=True).sum() / 1024**2

print(f"\nCURRENT MEMORY: {new_memory_mb:.2f} MB")


# ============================================================
# TASK 31 — Mean Billed_Engine_Hours by Vendor
# ============================================================

print("\n" + "=" * 70)
print("TASK 31 — Mean billed hours by Vendor")
print("=" * 70)

vendor_mean_hours = (
    df.groupby("Vendor", observed=True)["Billed_Engine_Hours"]
    .mean()
    .sort_values(ascending=False)
)

print(vendor_mean_hours)


# ============================================================
# TASK 32 — Maximum Engine_RPM by Equipment_Type
# ============================================================

print("\n" + "=" * 70)
print("TASK 32 — Maximum Engine RPM by Equipment")
print("=" * 70)

equipment_max_rpm = (
    df.groupby("Equipment_Type", observed=True)["Engine_RPM"]
    .max()
    .sort_values(ascending=False)
)

print(equipment_max_rpm)


# ============================================================
# TASK 33 — Expected idle fuel dictionary
# ============================================================

print("\n" + "=" * 70)
print("TASK 33 — Expected idle fuel thresholds")
print("=" * 70)

expected_idle_fuel = {"Tower_Crane": 2.0, "Heavy_Excavator": 2.0, "Concrete_Pump": 2.0}

print(expected_idle_fuel)


# ============================================================
# TASK 34 — Map idle threshold
# ============================================================

print("\n" + "=" * 70)
print("TASK 34 — Idle_Threshold_LPH")
print("=" * 70)

df["Idle_Threshold_LPH"] = (
    df["Equipment_Type"].map(expected_idle_fuel).astype("float32")
)

print(df[["Equipment_Type", "Idle_Threshold_LPH"]].head())


# ============================================================
# TASK 35 — Fuel_Variance
# ============================================================

print("\n" + "=" * 70)
print("TASK 35 — Fuel_Variance")
print("=" * 70)

df["Fuel_Variance"] = (df["Fuel_LPH"] - df["Idle_Threshold_LPH"]).astype("float32")

print(df["Fuel_Variance"].head())


# ============================================================
# TASK 36 — Absolute Fuel_Variance
# ============================================================

print("\n" + "=" * 70)
print("TASK 36 — Absolute Fuel_Variance")
print("=" * 70)

df["Fuel_Variance"] = np.abs(df["Fuel_Variance"])

print(df["Fuel_Variance"].head())


# ============================================================
# TASK 37 — Proper .loc approach
# ============================================================

print("\n" + "=" * 70)
print("TASK 37 — .loc vectorized update")
print("=" * 70)

print("""
Using .loc allows conditional dataframe updates
without chained-assignment / SettingWithCopyWarning.
""")


# ============================================================
# TASK 38 — Update negative Fuel_Variance using .loc
# ============================================================

print("\n" + "=" * 70)
print("TASK 38 — .loc update for negative variance")
print("=" * 70)

# Recalculate signed variance temporarily so the
# negative condition can be demonstrated.
signed_fuel_variance = df["Fuel_LPH"] - df["Idle_Threshold_LPH"]

negative_variance_mask = signed_fuel_variance < 0

# Store the magnitude for negative variances.
df.loc[negative_variance_mask, "Fuel_Variance"] = np.abs(
    signed_fuel_variance.loc[negative_variance_mask]
)

print("Negative Fuel_Variance rows updated:", negative_variance_mask.sum())


# ============================================================
# TASK 39 — Physics anomaly
# Engine_RPM > 1800 AND Fuel_LPH < 1.0
# ============================================================

print("\n" + "=" * 70)
print("TASK 39 — Physics_Anomaly")
print("=" * 70)

physics_condition = (df["Engine_RPM"] > 1800) & (df["Fuel_LPH"] < 1.0)

df["Physics_Anomaly"] = np.where(physics_condition, "Physics_Anomaly", "Normal")

print(df["Physics_Anomaly"].value_counts())


# ============================================================
# TASK 40 — Count Physics Anomalies
# ============================================================

print("\n" + "=" * 70)
print("TASK 40 — Physics Anomaly count")
print("=" * 70)

physics_anomaly_count = physics_condition.sum()

print("TOTAL PHYSICS ANOMALIES:", physics_anomaly_count)


# ============================================================
# TASK 41 — Tower Crane Physics Anomalies
# ============================================================

print("\n" + "=" * 70)
print("TASK 41 — Tower Crane Physics Anomalies")
print("=" * 70)

tower_crane_anomalies = df[
    (df["Equipment_Type"] == "Tower_Crane")
    & (df["Physics_Anomaly"] == "Physics_Anomaly")
].copy()

print("Tower Crane Physics Anomalies:", len(tower_crane_anomalies))

print(
    tower_crane_anomalies[
        [
            "Log_ID",
            "Vendor",
            "Equipment_Type",
            "Billed_Engine_Hours",
            "Billed_Amount_USD",
            "Engine_RPM",
            "Fuel_LPH",
        ]
    ].head(10)
)


# ============================================================
# TASK 42 — Total billing for Tower Crane anomalies
# ============================================================

print("\n" + "=" * 70)
print("TASK 42 — Billing associated with impossible subset")
print("=" * 70)

tower_anomaly_billing = tower_crane_anomalies["Billed_Amount_USD"].sum()

print(f"TOWER CRANE PHYSICS ANOMALY BILLING: " f"${tower_anomaly_billing:,.2f}")


# ============================================================
# TASK 43 — Sort chronologically
# ============================================================

print("\n" + "=" * 70)
print("TASK 43 — Chronological sorting")
print("=" * 70)

df = df.sort_values(by="Telemetry_Date").copy()

print("Earliest:", df["Telemetry_Date"].min())

print("Latest:", df["Telemetry_Date"].max())


# ============================================================
# TASK 44 — Reset index
# ============================================================

print("\n" + "=" * 70)
print("TASK 44 — Reset dataframe index")
print("=" * 70)

df.reset_index(drop=True, inplace=True)

print("Index starts at:", df.index.min())

print("Index ends at:", df.index.max())


# ============================================================
# TASK 45 — Rolling 3-day average Fuel_LPH by Vendor
# ============================================================

print("\n" + "=" * 70)
print("TASK 45 — 3-day rolling Fuel_LPH by Vendor")
print("=" * 70)

# Make sure the dataframe is sorted by Vendor and date
df = df.sort_values(["Vendor", "Telemetry_Date"]).copy()

# Set date as temporary index
temp = df.set_index("Telemetry_Date")

rolling_fuel = temp.groupby("Vendor")["Fuel_LPH"].rolling("3D", min_periods=1).mean()

# rolling_fuel has a MultiIndex:
# Vendor + Telemetry_Date
# Align it back to the original rows using Log_ID
#
# To avoid index alignment problems, calculate the rolling
# values group-by-group using transform-like logic.

df["Rolling_3D_Avg_Fuel_LPH"] = (
    df.groupby("Vendor", observed=True, group_keys=False)
    .apply(
        lambda group: (
            group.set_index("Telemetry_Date")["Fuel_LPH"]
            .rolling("3D", min_periods=1)
            .mean()
            .to_numpy()
        ),
        include_groups=False,
    )
    .explode()
    .astype("float32")
    .to_numpy()
)

print(df[["Vendor", "Telemetry_Date", "Fuel_LPH", "Rolling_3D_Avg_Fuel_LPH"]].head(10))


# ============================================================
# TASK 46 — Extract Encoded_GPS as standalone Series
# ============================================================

print("\n" + "=" * 70)
print("TASK 46 — Encoded_GPS Series")
print("=" * 70)

encoded_gps = df["Encoded_GPS"].copy()

print("Encoded_GPS Series type:", type(encoded_gps))

print(encoded_gps.head())


# ============================================================
# TASK 47 — Check unique Base64 string lengths
# ============================================================

print("\n" + "=" * 70)
print("TASK 47 — Base64 string length check")
print("=" * 70)

gps_lengths = encoded_gps.dropna().astype("string").str.len()

unique_gps_lengths = gps_lengths.value_counts().sort_index()

print("Unique Encoded_GPS lengths:")
print(unique_gps_lengths)

print("\nNumber of unique GPS string lengths:", gps_lengths.nunique())


# ============================================================
# TASK 48 — Export cleaned baseline to Parquet
# ============================================================

print("\n" + "=" * 70)
print("TASK 48 — Export to Parquet")
print("=" * 70)

try:

    df.to_parquet(PARQUET_FILE, index=False)

    print(f"Parquet file created: {PARQUET_FILE}")

except ImportError:

    print("\nPyArrow is not installed.")

    print("Install it in the VS Code terminal using:")

    print("pip install pyarrow")

    print("\nThen run this script again.")


# ============================================================
# TASK 49 — Assert exactly 1,000,000 rows
# ============================================================

print("\n" + "=" * 70)
print("TASK 49 — Row-count assertion")
print("=" * 70)

try:

    assert len(df) == 1_000_000

    print("PASS: Dataframe contains exactly 1,000,000 rows.")

except AssertionError:

    print("WARNING: Dataframe does NOT contain exactly " "1,000,000 rows.")

    print("Current row count:", len(df))

    print("This may indicate duplicate Log_ID records " "were removed during Task 18.")


# ============================================================
# TASK 50 — Assert zero null Billed_Amount_USD
# ============================================================

print("\n" + "=" * 70)
print("TASK 50 — Billing null assertion")
print("=" * 70)

billing_nulls = df["Billed_Amount_USD"].isnull().sum()

try:

    assert billing_nulls == 0

    print("PASS: Billed_Amount_USD contains zero null values.")

except AssertionError:

    print("WARNING: Billed_Amount_USD contains null values.")

    print("Null billing records:", billing_nulls)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 2 — FINAL SUMMARY")
print("=" * 70)

print(f"Original CSV: {CSV_FILE}")

print(f"Final dataframe rows: {len(df):,}")

print(f"Final dataframe columns: {len(df.columns)}")

print(f"Final memory usage: {new_memory_mb:.2f} MB")

print(f"Total Billed Amount: ${total_billing:,.2f}")

print(f"Negative/Zero Billing Records: " f"{len(negative_or_zero):,}")

print(f"Duplicate Log_IDs Removed: " f"{removed_rows:,}")

print(f"JSON Parsing Failures: " f"{json_failure_count:,}")

print(f"Physics Anomalies: " f"{physics_anomaly_count:,}")

print(f"Tower Crane Physics Anomalies: " f"{len(tower_crane_anomalies):,}")

print(f"Tower Crane Anomaly Billing: " f"${tower_anomaly_billing:,.2f}")

print("95th percentile from Day 1: 23.69505039 hours")

print("\nFiles generated:")
print(f"- {PARQUET_FILE}")

print("\n" + "=" * 70)
print("DAY 2 PROCESSING FINISHED")
print("=" * 70)


# ============================================================
# TASK 51 — Clear memory / prepare for Day 3
# ============================================================

print("\nTASK 51 — Memory cleanup")

del encoded_gps
del tower_crane_anomalies
del vendor_mean_hours
del equipment_max_rpm
del rolling_fuel

gc.collect()

print("Python objects cleaned from memory.")


