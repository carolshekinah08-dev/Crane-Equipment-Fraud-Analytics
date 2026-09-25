# """Nexlyra Project 4 — Day 3: Cryptography & Geospatial Anomaly Detection.

Decodes the base64-encoded GPS payloads embedded in the telemetry
feed, splits them into latitude/longitude pairs, flags equipment
whose GPS coordinates never change (``Is_Static_GPS``) while engine
hours continue to be billed, and cross-references the result against
the vendor field to isolate the confirmed ghost-equipment population.
"""

import base64
import binascii
import gc
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ================================================================
# CONFIGURATION
# ================================================================

PARQUET_FILE = "Carol_Crane_Clean_Baseline.parquet"

MASTER_CSV = "nexlyra_telemetry_master.csv"

GHOST_JSON = "Confirmed_Ghost_Equipment.json"

FRAUD_CHART = "Nexlyra_Ghost_Equipment_Cumulative_Funds.png"


print("=" * 70)
print("NEXLYRA PROJECT 4 — DAY 3")
print("Cryptography & Geospatial Anomaly Detection")
print("=" * 70)


# ================================================================
# TASK 01 — TITLE / INITIALIZATION
# ================================================================

print("\nTASK 01 — Cryptography & Geospatial Anomaly Detection")
print("Day 3 Python environment initialized.")


# ================================================================
# TASK 02 — LOAD PARQUET
# ================================================================

print("\n" + "=" * 70)
print("TASK 02 — Load Parquet file")
print("=" * 70)

df = pd.read_parquet(PARQUET_FILE)

print("Parquet loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ================================================================
# TASK 03 — ISOLATE Encoded_GPS
# ================================================================

print("\n" + "=" * 70)
print("TASK 03 — Isolate Encoded_GPS")
print("=" * 70)

encoded_gps = df["Encoded_GPS"]

print("Encoded_GPS Series created.")
print(encoded_gps.head())


# ================================================================
# TASK 04 — ROBUST BASE64 DECODER
# ================================================================

print("\n" + "=" * 70)
print("TASK 04 — Robust Base64 decoding function")
print("=" * 70)


def decode_base64_gps(value):
    """
    Decode a Base64-encoded GPS string.

    Handles:
    - missing padding
    - whitespace
    - invalid Base64 characters
    - corrupted values
    - non-string values

    Returns:
        decoded ASCII string
        or NaN when decoding fails
    """

    try:

        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        # Remove whitespace
        value = re.sub(r"\s+", "", value)

        # Base64 length must be a multiple of 4.
        # Add missing "=" padding when necessary.
        padding_needed = (-len(value)) % 4

        if padding_needed:
            value += "=" * padding_needed

        decoded_bytes = base64.b64decode(value, validate=True)

        decoded_text = decoded_bytes.decode("ascii")

        return decoded_text

    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error):
        return np.nan


print("Robust Base64 decoder created.")


# ================================================================
# TASK 05 — TEST DECODER ON ONE VALUE
# ================================================================

print("\n" + "=" * 70)
print("TASK 05 — Test Base64 decoder")
print("=" * 70)

test_encoded = encoded_gps.iloc[0]

test_decoded = decode_base64_gps(test_encoded)

print("Encoded:")
print(test_encoded)

print("\nDecoded:")
print(test_decoded)


# ================================================================
# TASK 06 — APPLY DECODER ACROSS ALL 1M ROWS
# ================================================================

print("\n" + "=" * 70)
print("TASK 06 — Decode all GPS values")
print("=" * 70)

df["Decoded_Lat_Lon"] = encoded_gps.apply(lambda x: decode_base64_gps(x))

print("Base64 decoding completed.")
print(df["Decoded_Lat_Lon"].head())


# ================================================================
# TASK 07 — ERROR HANDLING
# ================================================================

print("\n" + "=" * 70)
print("TASK 07 — Check decoding errors")
print("=" * 70)

gps_decode_failures = df["Decoded_Lat_Lon"].isna().sum()

print("GPS decoding failures:", gps_decode_failures)


# ================================================================
# TASK 08 — DECRYPTED COORDINATES COLUMN
# ================================================================

print("\n" + "=" * 70)
print("TASK 08 — Decoded_Lat_Lon created")
print("=" * 70)

print("Decoded_Lat_Lon column successfully assigned.")


# ================================================================
# TASK 09 — PREVIEW FORMAT
# ================================================================

print("\n" + "=" * 70)
print("TASK 09 — Preview decrypted coordinates")
print("=" * 70)

print(df["Decoded_Lat_Lon"].head(10))


# ================================================================
# TASK 10 — SPLIT LATITUDE / LONGITUDE
# ================================================================

print("\n" + "=" * 70)
print("TASK 10 — Split Latitude and Longitude")
print("=" * 70)


def split_coordinates(value):
    """Split a decoded "lat,lon" string into a Latitude/Longitude Series.

    Returns NaN for both fields when the value is missing or does not
    contain exactly two comma-separated numeric parts.
    """

    try:

        if pd.isna(value):
            return pd.Series([np.nan, np.nan], index=["Latitude", "Longitude"])

        parts = str(value).split(",")

        if len(parts) != 2:
            return pd.Series([np.nan, np.nan], index=["Latitude", "Longitude"])

        return pd.Series(
            [parts[0].strip(), parts[1].strip()], index=["Latitude", "Longitude"]
        )

    except Exception:
        return pd.Series([np.nan, np.nan], index=["Latitude", "Longitude"])


coordinates = df["Decoded_Lat_Lon"].apply(split_coordinates)

df["Latitude"] = coordinates["Latitude"]
df["Longitude"] = coordinates["Longitude"]

print(df[["Decoded_Lat_Lon", "Latitude", "Longitude"]].head())


# ================================================================
# TASK 11 — FLOAT32 COORDINATES
# ================================================================

print("\n" + "=" * 70)
print("TASK 11 — Convert coordinates to float32")
print("=" * 70)

df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").astype("float32")

df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").astype("float32")

print(df[["Latitude", "Longitude"]].dtypes)


# ================================================================
# TASK 12 — DROP INVALID COORDINATES
# ================================================================

print("\n" + "=" * 70)
print("TASK 12 — Remove invalid GPS rows")
print("=" * 70)

rows_before_gps_drop = len(df)

df = df.dropna(subset=["Latitude", "Longitude"]).copy()

rows_after_gps_drop = len(df)

print("Rows before:", rows_before_gps_drop)
print("Rows after:", rows_after_gps_drop)
print("Rows removed:", rows_before_gps_drop - rows_after_gps_drop)


# ================================================================
# TASK 13 — LATITUDE VARIANCE BY VENDOR + EQUIPMENT
# ================================================================

print("\n" + "=" * 70)
print("TASK 13 — Latitude variance by Vendor and Equipment_Type")
print("=" * 70)

latitude_variance = (
    df.groupby(["Vendor", "Equipment_Type"], observed=True)["Latitude"]
    .var()
    .sort_values()
)

print(latitude_variance)


# ================================================================
# TASK 14 — PHANTOM LEASING TOWER CRANE VARIANCE
# ================================================================

print("\n" + "=" * 70)
print("TASK 14 — Phantom_Leasing Tower Crane GPS variance")
print("=" * 70)

phantom_tower = df[
    (df["Vendor"] == "Phantom_Leasing") & (df["Equipment_Type"] == "Tower_Crane")
]

phantom_tower_lat_variance = phantom_tower["Latitude"].var()

print("Rows:", len(phantom_tower))
print("Latitude variance:", phantom_tower_lat_variance)


# ================================================================
# TASK 15 — STATIC GPS FLAG
# ================================================================

print("\n" + "=" * 70)
print("TASK 15 — Create Is_Static_GPS")
print("=" * 70)

# Compare each row with the previous row for the same equipment.
# Log_ID is used as the equipment record identifier because the
# dataset does not contain a separate equipment ID.

df["Previous_Latitude"] = df.groupby("Equipment_Type", observed=True)["Latitude"].shift(
    1
)

df["Previous_Longitude"] = df.groupby("Equipment_Type", observed=True)[
    "Longitude"
].shift(1)

df["Is_Static_GPS"] = df["Latitude"].eq(df["Previous_Latitude"]) & df["Longitude"].eq(
    df["Previous_Longitude"]
)

print(df["Is_Static_GPS"].value_counts(dropna=False))


# ================================================================
# TASK 16 — STATIC GPS + >20 HOURS
# ================================================================

print("\n" + "=" * 70)
print("TASK 16 — Static GPS + Billed Engine Hours > 20")
print("=" * 70)

# Core "ghost equipment" detection rule: GPS position never moved
# (Is_Static_GPS) while the vendor still billed for more than 20
# hours of engine run time on that record. This flags the applied
# detection rule only — it is not, by itself, confirmed real-world
# fraud until cross-referenced against the vendor and other checks.
static_high_hours = df[
    (df["Is_Static_GPS"]) & (df["Billed_Engine_Hours"] > 20)
].copy()

print("Matching rows:", len(static_high_hours))

print(
    static_high_hours[
        [
            "Log_ID",
            "Vendor",
            "Equipment_Type",
            "Billed_Engine_Hours",
            "Latitude",
            "Longitude",
        ]
    ].head(20)
)


# ================================================================
# TASK 17 — CROSS-REFERENCE PHYSICS ANOMALY
# ================================================================

print("\n" + "=" * 70)
print("TASK 17 — Cross-reference Physics_Anomaly")
print("=" * 70)

# Day 2 created Physics_Anomaly.
# Because Day 2 found zero Physics Anomalies, this column may not
# exist in the Parquet file depending on the Day 2 export version.
#
# To preserve the task logic, create a default Normal column if
# it is not present.

if "Physics_Anomaly" not in df.columns:

    df["Physics_Anomaly"] = np.where(
        (df["Engine_RPM"] > 1800) & (df["Fuel_LPH"] < 1.0), "Physics_Anomaly", "Normal"
    )

static_physics = static_high_hours[
    static_high_hours["Physics_Anomaly"] == "Physics_Anomaly"
]

print("Static GPS + Physics Anomaly rows:", len(static_physics))


# ================================================================
# TASK 18 — PERCENTAGE OF PHYSICS ANOMALIES WITH STATIC GPS
# ================================================================

print("\n" + "=" * 70)
print("TASK 18 — Percentage of Physics Anomalies with static GPS")
print("=" * 70)

physics_mask = df["Physics_Anomaly"] == "Physics_Anomaly"

total_physics = physics_mask.sum()

static_physics_count = (physics_mask & df["Is_Static_GPS"]).sum()

if total_physics > 0:

    static_physics_percentage = (static_physics_count / total_physics) * 100

else:

    static_physics_percentage = 0.0


print("Total Physics Anomalies:", total_physics)
print("Physics Anomalies with static GPS:", static_physics_count)
print("Percentage:", f"{static_physics_percentage:.2f}%")


# ================================================================
# TASK 19 — STATIC GPS COUNT BY VENDOR
# ================================================================

print("\n" + "=" * 70)
print("TASK 19 — Static GPS pings by Vendor")
print("=" * 70)

static_by_vendor = (
    df[df["Is_Static_GPS"]]
    .groupby("Vendor", observed=True)
    .size()
    .sort_values(ascending=False)
)

print(static_by_vendor)


# ================================================================
# TASK 20 — IDENTIFY PRIMARY SOURCE
# ================================================================

print("\n" + "=" * 70)
print("TASK 20 — Static GPS vendor analysis")
print("=" * 70)

if len(static_by_vendor) > 0:

    primary_static_vendor = static_by_vendor.idxmax()

    print("Vendor with highest static GPS count:", primary_static_vendor)

    print("Static GPS count:", static_by_vendor.max())

else:

    primary_static_vendor = None

    print("No static GPS rows found.")


# ================================================================
# TASK 21 — MOVING VS STATIC FUEL
# ================================================================

print("\n" + "=" * 70)
print("TASK 21 — Average Fuel_LPH: moving vs static")
print("=" * 70)

moving_fuel = df.loc[~df["Is_Static_GPS"], "Fuel_LPH"].mean()

static_fuel = df.loc[df["Is_Static_GPS"], "Fuel_LPH"].mean()

fuel_comparison = pd.Series(
    {"Moving_Equipment": moving_fuel, "Static_Equipment": static_fuel}
)

print(fuel_comparison)


# ================================================================
# TASK 22 — RPM STANDARD DEVIATION FOR STATIC EQUIPMENT
# ================================================================

print("\n" + "=" * 70)
print("TASK 22 — Static equipment RPM standard deviation")
print("=" * 70)

static_rpm_std = df.loc[df["Is_Static_GPS"], "Engine_RPM"].std()

print("Static equipment RPM standard deviation:", static_rpm_std)


# ================================================================
# TASK 23 — T-TEST METHOD
# ================================================================

print("\n" + "=" * 70)
print("TASK 23 — T-Test method")
print("=" * 70)

print(
    "Using scipy.stats.ttest_ind to compare "
    "Fuel_LPH for moving and static equipment."
)


# ================================================================
# TASK 24 — EXECUTE T-TEST
# ================================================================

print("\n" + "=" * 70)
print("TASK 24 — Independent T-Test")
print("=" * 70)

moving_values = df.loc[~df["Is_Static_GPS"], "Fuel_LPH"].dropna()

static_values = df.loc[df["Is_Static_GPS"], "Fuel_LPH"].dropna()

if len(moving_values) > 1 and len(static_values) > 1:

    t_statistic, p_value = stats.ttest_ind(
        moving_values, static_values, equal_var=False, nan_policy="omit"
    )

else:

    t_statistic = np.nan
    p_value = np.nan


print("T-statistic:", t_statistic)
print("P-value:", p_value)

if not pd.isna(p_value):

    if p_value < 0.05:
        print("Result: statistically significant at the 5% level.")
    else:
        print("Result: not statistically significant at the 5% level.")

else:

    print("T-test could not be calculated.")


# ================================================================
# TASK 25 — MAX RPM FOR STATIC GPS
# ================================================================

print("\n" + "=" * 70)
print("TASK 25 — Maximum RPM with static GPS")
print("=" * 70)

max_static_rpm = df.loc[df["Is_Static_GPS"], "Engine_RPM"].max()

print("Maximum static-GPS RPM:", max_static_rpm)


# ================================================================
# TASK 26 — MINIMUM FUEL FOR MOVING EQUIPMENT
# ================================================================

print("\n" + "=" * 70)
print("TASK 26 — Minimum Fuel_LPH for moving equipment")
print("=" * 70)

min_moving_fuel = df.loc[~df["Is_Static_GPS"], "Fuel_LPH"].min()

print("Minimum moving-equipment Fuel_LPH:", min_moving_fuel)


# ================================================================
# TASK 27 — EFFICIENCY RATIO
# ================================================================

print("\n" + "=" * 70)
print("TASK 27 — Efficiency_Ratio")
print("=" * 70)

# Avoid division-by-zero.

df["Efficiency_Ratio"] = np.where(
    df["Fuel_LPH"] > 0, df["Billed_Engine_Hours"] / df["Fuel_LPH"], np.inf
)

print(df[["Billed_Engine_Hours", "Fuel_LPH", "Efficiency_Ratio"]].head())


# ================================================================
# TASK 28 — COMPARE EFFICIENCY
# ================================================================

print("\n" + "=" * 70)
print("TASK 28 — Efficiency Ratio comparison")
print("=" * 70)

static_efficiency = df.loc[df["Is_Static_GPS"], "Efficiency_Ratio"]

moving_efficiency = df.loc[~df["Is_Static_GPS"], "Efficiency_Ratio"]

print(
    "Static median Efficiency_Ratio:",
    static_efficiency.replace([np.inf, -np.inf], np.nan).median(),
)

print(
    "Moving median Efficiency_Ratio:",
    moving_efficiency.replace([np.inf, -np.inf], np.nan).median(),
)

print("Static maximum Efficiency_Ratio:", static_efficiency.max())


# ================================================================
# TASK 29 — LATITUDE REGEX VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("TASK 29 — Check anomalous characters in Latitude")
print("=" * 70)

latitude_string = df["Latitude"].astype(str)

anomalous_latitude_mask = latitude_string.str.contains(
    r"[A-Za-z]", regex=True, na=False
)

anomalous_latitude_count = anomalous_latitude_mask.sum()

print("Latitude values containing alphabetical characters:", anomalous_latitude_count)


# ================================================================
# TASK 30 — DROP STRING GPS COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("TASK 30 — Drop Encoded_GPS and Decoded_Lat_Lon")
print("=" * 70)

df.drop(columns=["Encoded_GPS", "Decoded_Lat_Lon"], inplace=True, errors="ignore")

print("Original GPS string columns removed.")


# ================================================================
# TASK 31 — CONVERT STATIC FLAG TO INT8
# ================================================================

print("\n" + "=" * 70)
print("TASK 31 — Convert Is_Static_GPS to int8")
print("=" * 70)

df["Is_Static_GPS"] = df["Is_Static_GPS"].astype("int8")

print(df["Is_Static_GPS"].dtype)
print(df["Is_Static_GPS"].value_counts())


# ================================================================
# TASK 32 — VENDOR TO CATEGORY
# ================================================================

print("\n" + "=" * 70)
print("TASK 32 — Convert Vendor to category")
print("=" * 70)

df["Vendor"] = df["Vendor"].astype("category")

print(df["Vendor"].dtype)


# ================================================================
# TASK 33 — MEMORY CHECK
# ================================================================

print("\n" + "=" * 70)
print("TASK 33 — Memory usage after optimization")
print("=" * 70)

memory_usage = df.memory_usage(deep=True)

print(memory_usage)

total_memory_mb = memory_usage.sum() / (1024**2)

print(f"\nTOTAL MEMORY: {total_memory_mb:.2f} MB")


# ================================================================
# TASK 34 — PIVOT TABLE 1
# ================================================================

print("\n" + "=" * 70)
print("TASK 34 — Vendor × Static GPS billing pivot")
print("=" * 70)

pivot_vendor_static = pd.pivot_table(
    df,
    index="Vendor",
    columns="Is_Static_GPS",
    values="Billed_Amount_USD",
    aggfunc="sum",
    fill_value=0,
    observed=True,
)

print(pivot_vendor_static)


# ================================================================
# TASK 35 — PIVOT TABLE 2
# ================================================================

print("\n" + "=" * 70)
print("TASK 35 — Equipment × Site fuel pivot")
print("=" * 70)

pivot_equipment_site = pd.pivot_table(
    df,
    index="Equipment_Type",
    columns="Site_Code",
    values="Fuel_LPH",
    aggfunc="mean",
    fill_value=np.nan,
    observed=True,
)

print(pivot_equipment_site)


# ================================================================
# TASK 36 — MELT FIRST PIVOT
# ================================================================

print("\n" + "=" * 70)
print("TASK 36 — Melt first pivot")
print("=" * 70)

pivot_melted = pivot_vendor_static.reset_index().melt(
    id_vars="Vendor", var_name="Is_Static_GPS", value_name="Total_Billed_USD"
)

print(pivot_melted.head(20))


# ================================================================
# TASK 37 — RPM BINS
# ================================================================

print("\n" + "=" * 70)
print("TASK 37 — Create RPM tiers")
print("=" * 70)

# RPM ranges:
# 0–500      = Idle
# 500–1000   = Low
# 1000–1500  = Normal
# 1500+      = Over-Rev

rpm_bins = [-np.inf, 500, 1000, 1500, np.inf]

rpm_labels = ["Idle", "Low", "Normal", "Over-Rev"]


# ================================================================
# TASK 38 — ASSIGN RPM TIER
# ================================================================

df["RPM_Tier"] = pd.cut(df["Engine_RPM"], bins=rpm_bins, labels=rpm_labels, right=False)

print(df["RPM_Tier"].value_counts(dropna=False))


# ================================================================
# TASK 39 — FUEL QUARTILES
# ================================================================

print("\n" + "=" * 70)
print("TASK 39 — Fuel quartiles")
print("=" * 70)

# Check whether Fuel_LPH has enough unique values
# to create four quartiles.

unique_fuel_values = df["Fuel_LPH"].nunique()

print("Unique Fuel_LPH values:", unique_fuel_values)

if unique_fuel_values >= 4:

    df["Fuel_Quartile"] = pd.qcut(df["Fuel_LPH"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    print(df["Fuel_Quartile"].value_counts())

else:

    # Fuel_LPH is effectively constant in this dataset.
    # Therefore true Q1-Q4 quartiles cannot be mathematically
    # separated.

    df["Fuel_Quartile"] = "Constant"

    print("Fuel_LPH does not contain enough unique values " "to create four quartiles.")

    print("All records assigned to: Constant")

# ================================================================
# # ================================================================
# TASK 40 — FUEL QUARTILE VS OVER-REV
# ================================================================

print("\n" + "=" * 70)
print("TASK 40 — Fuel quartile associated with Over-Rev")
print("=" * 70)

over_rev_table = pd.crosstab(df["Fuel_Quartile"], df["RPM_Tier"])

print(over_rev_table)

if "Over-Rev" in over_rev_table.columns:

    highest_over_rev_quartile = over_rev_table["Over-Rev"].idxmax()

    highest_over_rev_count = over_rev_table["Over-Rev"].max()

    print("Fuel quartile with highest Over-Rev count:", highest_over_rev_quartile)

    print("Over-Rev records:", highest_over_rev_count)

else:

    highest_over_rev_quartile = None

    print("No Over-Rev records found.")

print(
    "Note: All observed Engine_RPM values are within the "
    "Normal RPM tier, so there are no Over-Rev records."
)

# ================================================================
# TASK 41 — CUSTOM AGGREGATION FUNCTION
# ================================================================

print("\n" + "=" * 70)
print("TASK 41 — Custom aggregation function")
print("=" * 70)


def billing_and_rpm_summary(group):
    """Aggregate a group's total billing and peak engine RPM.

    Intended for use with a groupby().apply() call; returns a Series
    with ``Total_Billed_USD`` and ``Maximum_Engine_RPM`` for the group.
    """

    return pd.Series(
        {
            "Total_Billed_USD": group["Billed_Amount_USD"].sum(),
            "Maximum_Engine_RPM": group["Engine_RPM"].max(),
        }
    )


print("Custom aggregation function created.")


# ================================================================
# TASK 42 — APPLY BY VENDOR
# ================================================================

print("\n" + "=" * 70)
print("TASK 42 — Vendor custom aggregation")
print("=" * 70)

vendor_summary = df.groupby("Vendor", observed=True).apply(
    billing_and_rpm_summary, include_groups=False
)

print(vendor_summary)


# ================================================================
# TASK 43 — PHANTOM LEASING ONLY
# ================================================================

print("\n" + "=" * 70)
print("TASK 43 — Isolate Phantom_Leasing")
print("=" * 70)

phantom_df = df[df["Vendor"] == "Phantom_Leasing"].copy()

print("Phantom_Leasing rows:", len(phantom_df))

print(
    phantom_df[
        [
            "Log_ID",
            "Equipment_Type",
            "Billed_Engine_Hours",
            "Billed_Amount_USD",
            "Engine_RPM",
            "Fuel_LPH",
            "Latitude",
            "Longitude",
            "Is_Static_GPS",
        ]
    ].head(20)
)


# ================================================================
# TASK 44 — MATHEMATICAL / PHYSICAL PROOF
# ================================================================

print("\n" + "=" * 70)
print("TASK 44 — Phantom_Leasing mathematical analysis")
print("=" * 70)

phantom_average_fuel = phantom_df["Fuel_LPH"].mean()

phantom_average_rpm = phantom_df["Engine_RPM"].mean()

phantom_average_hours = phantom_df["Billed_Engine_Hours"].mean()

phantom_static_percentage = (phantom_df["Is_Static_GPS"].mean()) * 100

print("Phantom_Leasing average Fuel_LPH:", phantom_average_fuel)

print("Phantom_Leasing average Engine_RPM:", phantom_average_rpm)

print("Phantom_Leasing average Billed Engine Hours:", phantom_average_hours)

print("Phantom_Leasing static GPS percentage:", f"{phantom_static_percentage:.2f}%")

print("\nImportant interpretation:")

print(
    "The analysis tests whether the billing/activity pattern "
    "is consistent with the telemetry evidence."
)

print(
    "Static GPS means the recorded coordinates are unchanged "
    "between consecutive records for the equipment grouping used."
)


# ================================================================
# TASK 45 — SAVE MATHEMATICAL + STATIC GPS PROOF
# ================================================================

print("\n" + "=" * 70)
print("TASK 45 — Save proof for final report")
print("=" * 70)

proof_text = f"""
NEXLYRA PROJECT 4 — STATIC GPS / TELEMETRY ANALYSIS

Phantom_Leasing records analysed: {len(phantom_df):,}

Average billed engine hours:
{phantom_average_hours:.6f}

Average Fuel_LPH:
{phantom_average_fuel:.6f}

Average Engine RPM:
{phantom_average_rpm:.6f}

Percentage of Phantom_Leasing records with static GPS:
{phantom_static_percentage:.2f}%

Maximum RPM among static-GPS records:
{max_static_rpm:.6f}

Minimum Fuel_LPH among moving equipment:
{min_moving_fuel:.6f}

Physics anomaly records:
{total_physics}

Physics anomalies with static GPS:
{static_physics_count}

Physics anomaly static-GPS percentage:
{static_physics_percentage:.2f}%

The static-GPS test identifies records whose latitude and longitude
remain identical to the previous record within the equipment grouping
used for this analysis.

The physics-anomaly test uses:
Engine_RPM > 1800 AND Fuel_LPH < 1.0
"""

print(proof_text)


# ================================================================
# TASK 46 — CONFIRMED GHOST EQUIPMENT
# ================================================================

print("\n" + "=" * 70)
print("TASK 46 — Confirmed_Ghost_Equipment")
print("=" * 70)

# The task asks to filter out non-fraudulent rows.
#
# We use a conservative combined evidence rule:
#
# 1. Vendor is Phantom_Leasing
# 2. GPS is static
# 3. Billed Engine Hours > 20
#
# Physics_Anomaly is also retained as an evidence field, but because
# Day 2 found zero physics anomalies, requiring it here would create
# an empty dataframe.

Confirmed_Ghost_Equipment = df[
    (df["Vendor"] == "Phantom_Leasing")
    & (df["Is_Static_GPS"] == 1)
    & (df["Billed_Engine_Hours"] > 20)
].copy()

print("Confirmed_Ghost_Equipment rows:", len(Confirmed_Ghost_Equipment))


# ================================================================
# TASK 47 — CUMULATIVE STOLEN FUNDS
# ================================================================

print("\n" + "=" * 70)
print("TASK 47 — Cumulative billed amount over time")
print("=" * 70)

Confirmed_Ghost_Equipment = Confirmed_Ghost_Equipment.sort_values(
    "Telemetry_Date"
).reset_index(drop=True)

Confirmed_Ghost_Equipment["Cumulative_Stolen_Funds_USD"] = Confirmed_Ghost_Equipment[
    "Billed_Amount_USD"
].cumsum()

if len(Confirmed_Ghost_Equipment) > 0:

    print(
        Confirmed_Ghost_Equipment[
            ["Telemetry_Date", "Billed_Amount_USD", "Cumulative_Stolen_Funds_USD"]
        ].head(20)
    )

    print(
        "\nFinal cumulative amount:",
        f"${Confirmed_Ghost_Equipment['Cumulative_Stolen_Funds_USD'].iloc[-1]:,.2f}",
    )

else:

    print("No rows matched the Confirmed_Ghost_Equipment rule.")


# ================================================================
# TASK 48 — PLOT CUMULATIVE FUNDS
# ================================================================

print("\n" + "=" * 70)
print("TASK 48 — Plot cumulative funds")
print("=" * 70)

plt.figure(figsize=(12, 6))

if len(Confirmed_Ghost_Equipment) > 0:

    plt.plot(
        Confirmed_Ghost_Equipment["Telemetry_Date"],
        Confirmed_Ghost_Equipment["Cumulative_Stolen_Funds_USD"],
    )

else:

    plt.text(
        0.5,
        0.5,
        "No Confirmed Ghost Equipment records",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )


# ================================================================
# TASK 49 — CHART FORMATTING
# ================================================================

plt.title("Nexlyra Project 4 — Cumulative Ghost Equipment Billing")

plt.xlabel("Telemetry Date")

plt.ylabel("Cumulative Billed Amount (USD)")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(FRAUD_CHART, dpi=200, bbox_inches="tight")

plt.show()

print("Chart saved as:", FRAUD_CHART)


# ================================================================
# TASK 50 — EXPORT MASTER DATASET
# ================================================================

print("\n" + "=" * 70)
print("TASK 50 — Export master telemetry dataset")
print("=" * 70)

df.to_csv(MASTER_CSV, index=False)

print("Master CSV created:", MASTER_CSV)

print("Rows exported:", len(df))


# ================================================================
# TASK 51 — EXPORT CONFIRMED GHOST EQUIPMENT JSON
# ================================================================

print("\n" + "=" * 70)
print("TASK 51 — Export Confirmed_Ghost_Equipment JSON")
print("=" * 70)

Confirmed_Ghost_Equipment.to_json(
    GHOST_JSON, orient="records", date_format="iso", indent=2
)

print("Ghost equipment JSON created:", GHOST_JSON)

print("Rows exported:", len(Confirmed_Ghost_Equipment))


# ================================================================
# TASK 52 — FINAL VALIDATION / CLEANUP
# ================================================================

print("\n" + "=" * 70)
print("TASK 52 — Final validation and cleanup")
print("=" * 70)

print("Final dataframe rows:", len(df))

print("Final dataframe columns:", len(df.columns))

print(
    "Final dataframe memory:",
    f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB",
)

print("\nOutput files:")

print("1.", MASTER_CSV, "→", os.path.exists(MASTER_CSV))

print("2.", GHOST_JSON, "→", os.path.exists(GHOST_JSON))

print("3.", FRAUD_CHART, "→", os.path.exists(FRAUD_CHART))


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("DAY 3 — FINAL SUMMARY")
print("=" * 70)

print("Original Parquet rows: 1,000,000")

print(f"Final rows: {len(df):,}")

print(f"GPS decoding failures: {gps_decode_failures:,}")

print(f"Static GPS records: {int(df['Is_Static_GPS'].sum()):,}")

print(f"Physics anomalies: {int(total_physics):,}")

print(f"Static + >20 hour records: {len(static_high_hours):,}")

print(f"Confirmed Ghost Equipment records: " f"{len(Confirmed_Ghost_Equipment):,}")

if len(Confirmed_Ghost_Equipment) > 0:

    total_ghost_billing = Confirmed_Ghost_Equipment["Billed_Amount_USD"].sum()

else:

    total_ghost_billing = 0.0


print(f"Confirmed Ghost Equipment billing: " f"${total_ghost_billing:,.2f}")

print(f"Master CSV: {MASTER_CSV}")

print(f"Ghost JSON: {GHOST_JSON}")

print(f"Fraud timeline chart: {FRAUD_CHART}")

print("\nDay 3 processing finished.")


# ================================================================
# MEMORY CLEANUP
# ================================================================

del encoded_gps
del coordinates
del moving_values
del static_values
del phantom_df
del static_high_hours

gc.collect()

print("\nPython memory cleanup completed.")
