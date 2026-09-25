# """Nexlyra Project 4 — Day 4: Advanced SQL & Relational Data Modeling.

Builds a normalized SQLite database from the master telemetry CSV,
runs the forensic audit queries (ghost-equipment detection, static
GPS cross-checks, financial exposure by vendor), and exports the
executive-summary and physics-violation result sets used in later
reporting.
"""

import csv
import sqlite3
import time
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "nexlyra_telemetry_master.csv"
DB_FILE = BASE_DIR / "nexlyra_telematics.db"

EXECUTIVE_CSV = BASE_DIR / "executive_summary.csv"
PHYSICS_CSV = BASE_DIR / "physics_violations.csv"
AUDIT_SQL_FILE = BASE_DIR / "telematics_audit.sql"

# ================================================================
# DAY 4 HEADER
# ================================================================

print("=" * 70)
print("NEXLYRA PROJECT 4 — DAY 4")
print("Advanced SQL & Relational Data Modeling")
print("=" * 70)

# ================================================================
# TASK 01 — Advanced SQL & Relational Data Modeling
# ================================================================

print("\n" + "=" * 70)
print("TASK 01 — Advanced SQL & Relational Data Modeling")
print("=" * 70)

print("Day 4 focuses on SQLite relational modeling and advanced SQL.")


# ================================================================
# TASK 02 — Initialize SQLite database
# ================================================================

print("\n" + "=" * 70)
print("TASK 02 — Initialize SQLite database")
print("=" * 70)

if DB_FILE.exists():
    print("Existing database found.")
    print("It will be replaced for a clean Day 4 run.")
    DB_FILE.unlink()

conn = sqlite3.connect(DB_FILE)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")

print(f"SQLite database created: {DB_FILE.name}")


# ================================================================
# HELPER FUNCTIONS
# ================================================================


def table_columns(cursor, table_name):
    """Return SQLite column names for a table."""
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    return [row[1] for row in cursor.fetchall()]


def quote_identifier(name):
    """Safely quote a SQLite identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_value(value):
    """Convert Python value into a SQL literal for SQL export."""
    if value is None:
        return "NULL"

    text = str(value)

    if text.strip() == "":
        return "NULL"

    # Try numeric
    try:
        float(text)
        return text
    except ValueError:
        pass

    return "'" + text.replace("'", "''") + "'"


def run_query(title, query, params=(), fetch=True, limit=20):
    """Execute and print SQL query results."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(query.strip())

    cur = conn.cursor()
    cur.execute(query, params)

    if fetch:
        rows = cur.fetchmany(limit)
        for row in rows:
            print(row)

        if len(rows) == limit:
            print(f"... showing first {limit} rows")

        return rows

    return None


def export_query_to_csv(query, output_file, params=()):
    """Export SQL query result to CSV."""
    cur = conn.cursor()
    cur.execute(query, params)

    rows = cur.fetchall()
    headers = [description[0] for description in cur.description]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Exported: {output_file.name}")
    print(f"Rows exported: {len(rows)}")


# ================================================================
# TASK 03 — Import master CSV
# ================================================================

print("\n" + "=" * 70)
print("TASK 03 — Import nexlyra_telemetry_master.csv")
print("=" * 70)

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: {CSV_FILE.name} was not found.\n"
        f"Place nexlyra_telemetry_master.csv in:\n{BASE_DIR}"
    )

print(f"Input CSV: {CSV_FILE.name}")

# ------------------------------------------------
# Read CSV header
# ------------------------------------------------

with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)

print(f"Columns detected: {len(headers)}")
print(headers)

# ------------------------------------------------
# Create SQL-friendly column names
# ------------------------------------------------

# Preserve expected names from Day 3.
# All columns are initially TEXT so SQLite import
# is robust against mixed CSV representations.
column_definitions = []

for col in headers:
    clean_col = col.strip()

    if clean_col == "Log_ID":
        column_definitions.append('"Log_ID" TEXT PRIMARY KEY')
    else:
        column_definitions.append(f'"{clean_col}" TEXT')

create_fact_sql = f"""
CREATE TABLE Telemetry_Fact_Table (
    {", ".join(column_definitions)}
);
"""

conn.execute(create_fact_sql)

print("Telemetry_Fact_Table created.")

# ------------------------------------------------
# Import using sqlite executemany in chunks
# ------------------------------------------------

placeholders = ",".join(["?"] * len(headers))

insert_sql = f"""
INSERT INTO Telemetry_Fact_Table
({",".join(quote_identifier(h) for h in headers)})
VALUES ({placeholders})
"""

CHUNK_SIZE = 5000

row_count = 0

with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:

    reader = csv.reader(f)
    next(reader)

    batch = []

    for row in reader:
        batch.append(row)

        if len(batch) >= CHUNK_SIZE:
            conn.executemany(insert_sql, batch)
            row_count += len(batch)

            if row_count % 100000 == 0:
                print(f"Imported {row_count:,} rows...")

            batch.clear()

    if batch:
        conn.executemany(insert_sql, batch)
        row_count += len(batch)

conn.commit()

print(f"Total rows imported: {row_count:,}")


# ================================================================
# TASK 04 — Dim_Vendor
# ================================================================

print("\n" + "=" * 70)
print("TASK 04 — Create Dim_Vendor")
print("=" * 70)

conn.execute("""
CREATE TABLE Dim_Vendor (
    Vendor_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Vendor TEXT UNIQUE NOT NULL
);
""")

conn.execute("""
INSERT INTO Dim_Vendor (Vendor)
SELECT DISTINCT Vendor
FROM Telemetry_Fact_Table
WHERE Vendor IS NOT NULL
AND TRIM(Vendor) <> '';
""")

conn.commit()

print("Dim_Vendor created.")
run_query(
    "Vendor dimension:",
    """
    SELECT Vendor_ID, Vendor
    FROM Dim_Vendor
    ORDER BY Vendor_ID;
    """,
)


# ================================================================
# TASK 05 — Dim_Equipment
# ================================================================

print("\n" + "=" * 70)
print("TASK 05 — Create Dim_Equipment")
print("=" * 70)

conn.execute("""
CREATE TABLE Dim_Equipment (
    Equipment_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Equipment_Type TEXT UNIQUE NOT NULL,
    Standard_Rental_Rate_USD REAL
);
""")

# Rates from Day 1 lookup
equipment_rates = {
    "Heavy_Excavator": 250.0,
    "Concrete_Pump": 200.0,
    "Tower_Crane": 350.0,
}

for equipment, rate in equipment_rates.items():
    conn.execute(
        """
        INSERT OR IGNORE INTO Dim_Equipment
        (Equipment_Type, Standard_Rental_Rate_USD)
        VALUES (?, ?)
        """,
        (equipment, rate),
    )

conn.commit()

print("Dim_Equipment created.")

run_query(
    "Equipment dimension:",
    """
    SELECT *
    FROM Dim_Equipment
    ORDER BY Equipment_ID;
    """,
)


# ================================================================
# TASK 06 — Dim_Site
# ================================================================

print("\n" + "=" * 70)
print("TASK 06 — Create Dim_Site")
print("=" * 70)

conn.execute("""
CREATE TABLE Dim_Site (
    Site_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Site_Code TEXT UNIQUE NOT NULL,
    Project_Zone TEXT
);
""")

site_zone_map = {
    "NEX-BRIDGE-09": "Bridge",
    "NEX-MALL-22": "Mall",
    "NEX-STADIUM-05": "Stadium",
    "NEX-TOWER-01": "Tower",
}

for site, zone in site_zone_map.items():
    conn.execute(
        """
        INSERT OR IGNORE INTO Dim_Site
        (Site_Code, Project_Zone)
        VALUES (?, ?)
        """,
        (site, zone),
    )

conn.commit()

print("Dim_Site created.")

run_query(
    "Site dimension:",
    """
    SELECT *
    FROM Dim_Site
    ORDER BY Site_ID;
    """,
)


# ================================================================
# TASK 07 — Primary Key
# ================================================================

print("\n" + "=" * 70)
print("TASK 07 — Primary Key")
print("=" * 70)

print("""
SQLite does not support adding a PRIMARY KEY to an existing
table using ALTER TABLE.

Therefore, the Telemetry_Fact_Table was created with:

    Log_ID TEXT PRIMARY KEY

during table creation.
""")

alter_statement = """
-- SQLite limitation:
-- ALTER TABLE cannot add a PRIMARY KEY to an existing table.
-- The primary key was therefore defined during CREATE TABLE.

-- Requested conceptual statement:
-- ALTER TABLE Telemetry_Fact_Table
-- ADD PRIMARY KEY (Log_ID);
"""

print(alter_statement)


# ================================================================
# TASK 08 — Create indexes
# ================================================================

print("\n" + "=" * 70)
print("TASK 08 — Create indexes")
print("=" * 70)

conn.execute("""
CREATE INDEX idx_fact_vendor
ON Telemetry_Fact_Table (Vendor);
""")

conn.execute("""
CREATE INDEX idx_fact_telemetry_date
ON Telemetry_Fact_Table (Telemetry_Date);
""")

conn.commit()

print("Index created: idx_fact_vendor")
print("Index created: idx_fact_telemetry_date")


# ================================================================
# TASK 09 — EXPLAIN QUERY PLAN explanation
# ================================================================

print("\n" + "=" * 70)
print("TASK 09 — EXPLAIN QUERY PLAN")
print("=" * 70)

print("""
EXPLAIN QUERY PLAN shows how SQLite intends to execute a query.

It can reveal whether SQLite is:
- scanning the entire table
- using an index
- searching a table by a primary key
- performing a temporary sort

Indexes can reduce the amount of data SQLite has to scan.
""")


# ================================================================
# TASK 10 — EXPLAIN QUERY PLAN JOIN
# ================================================================

print("\n" + "=" * 70)
print("TASK 10 — EXPLAIN QUERY PLAN on Vendor JOIN")
print("=" * 70)

query_plan = """
EXPLAIN QUERY PLAN
SELECT
    f.Log_ID,
    f.Vendor,
    v.Vendor_ID
FROM Telemetry_Fact_Table f
JOIN Dim_Vendor v
    ON f.Vendor = v.Vendor;
"""

run_query("Query execution plan:", query_plan)


# ================================================================
# TASK 11 — SUM + GROUP BY
# ================================================================

print("\n" + "=" * 70)
print("TASK 11 — Total billed USD per Vendor")
print("=" * 70)

vendor_billing_query = """
SELECT
    Vendor,
    SUM(CAST(Billed_Amount_USD AS REAL)) AS Total_Billed_USD
FROM Telemetry_Fact_Table
GROUP BY Vendor
ORDER BY Total_Billed_USD DESC;
"""

run_query("Vendor billing:", vendor_billing_query)


# ================================================================
# TASK 12 — AVG Fuel_LPH
# ================================================================

print("\n" + "=" * 70)
print("TASK 12 — Average Fuel_LPH per Equipment_Type")
print("=" * 70)

equipment_fuel_query = """
SELECT
    Equipment_Type,
    AVG(CAST(Fuel_LPH AS REAL)) AS Average_Fuel_LPH
FROM Telemetry_Fact_Table
GROUP BY Equipment_Type
ORDER BY Average_Fuel_LPH DESC;
"""

run_query("Average fuel by equipment:", equipment_fuel_query)


# ================================================================
# TASK 13 — CASE WHEN
# ================================================================

print("\n" + "=" * 70)
print("TASK 13 — CASE WHEN High_Suspicion")
print("=" * 70)

case_query = """
SELECT
    Log_ID,
    Vendor,
    Equipment_Type,
    Fuel_LPH,
    CASE
        WHEN CAST(Fuel_LPH AS REAL) < 0.5
        THEN 'High_Suspicion'
        ELSE 'Normal'
    END AS Suspicion_Category
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Fuel suspicion classification:", case_query)


# ================================================================
# TASK 14 — WINDOW FUNCTION
# ================================================================

print("\n" + "=" * 70)
print("TASK 14 — Window Function OVER(PARTITION BY Vendor)")
print("=" * 70)

window_query = """
SELECT
    Log_ID,
    Vendor,
    Billed_Amount_USD,
    SUM(CAST(Billed_Amount_USD AS REAL))
        OVER (PARTITION BY Vendor)
        AS Vendor_Total_Billing
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Vendor window function:", window_query)


# ================================================================
# TASK 15 — Running total per vendor
# ================================================================

print("\n" + "=" * 70)
print("TASK 15 — Running total of Billed_Amount_USD per Vendor")
print("=" * 70)

running_total_query = """
SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    Billed_Amount_USD,
    SUM(CAST(Billed_Amount_USD AS REAL))
        OVER (
            PARTITION BY Vendor
            ORDER BY Telemetry_Date, Log_ID
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ) AS Running_Total_USD
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Running billing total:", running_total_query)


# ================================================================
# TASK 16 — LAG()
# ================================================================

print("\n" + "=" * 70)
print("TASK 16 — LAG() Phantom_Leasing time difference")
print("=" * 70)

lag_query = """
WITH Phantom AS (
    SELECT
        Log_ID,
        Vendor,
        Telemetry_Date,
        LAG(Telemetry_Date)
            OVER (
                PARTITION BY Vendor
                ORDER BY Telemetry_Date, Log_ID
            ) AS Previous_Timestamp
    FROM Telemetry_Fact_Table
    WHERE Vendor = 'Phantom_Leasing'
)
SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    Previous_Timestamp,
    ROUND(
        (
            julianday(Telemetry_Date)
            - julianday(Previous_Timestamp)
        ) * 24 * 60,
        2
    ) AS Minutes_Since_Previous
FROM Phantom
LIMIT 20;
"""

run_query("LAG time difference:", lag_query)


# ================================================================
# TASK 17 — LEAD()
# ================================================================

print("\n" + "=" * 70)
print("TASK 17 — LEAD() next Engine_RPM")
print("=" * 70)

lead_query = """
SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    Engine_RPM,
    LEAD(Engine_RPM)
        OVER (
            PARTITION BY Vendor
            ORDER BY Telemetry_Date, Log_ID
        ) AS Next_Engine_RPM
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("LEAD RPM comparison:", lead_query)


# ================================================================
# TASK 18 — Static_Activity CTE
# ================================================================

print("\n" + "=" * 70)
print("TASK 18 — Static_Activity CTE")
print("=" * 70)

static_cte = """
WITH Static_Activity AS (
    SELECT *
    FROM Telemetry_Fact_Table
    WHERE CAST(Is_Static_GPS AS INTEGER) = 1
)
SELECT COUNT(*)
FROM Static_Activity;
"""

run_query("Static activity count:", static_cte)


# ================================================================
# TASK 19 — Top 10 static billing
# ================================================================

print("\n" + "=" * 70)
print("TASK 19 — Top 10 highest-billing static logs")
print("=" * 70)

top_static_query = """
WITH Static_Activity AS (
    SELECT *
    FROM Telemetry_Fact_Table
    WHERE CAST(Is_Static_GPS AS INTEGER) = 1
)
SELECT
    Log_ID,
    Vendor,
    Equipment_Type,
    Telemetry_Date,
    Billed_Engine_Hours,
    Billed_Amount_USD
FROM Static_Activity
ORDER BY CAST(Billed_Amount_USD AS REAL) DESC
LIMIT 10;
"""

run_query("Top static billing logs:", top_static_query, limit=10)


# ================================================================
# TASK 20 — Physics variance CTE
# ================================================================

print("\n" + "=" * 70)
print("TASK 20 — RPM/Fuel variance CTE")
print("=" * 70)

physics_cte = """
WITH Physics_Variance AS (
    SELECT
        Log_ID,
        Vendor,
        Equipment_Type,
        Engine_RPM,
        Fuel_LPH,
        CAST(Engine_RPM AS REAL)
        -
        CAST(Fuel_LPH AS REAL)
        AS RPM_Fuel_Variance
    FROM Telemetry_Fact_Table
)
SELECT *
FROM Physics_Variance
LIMIT 20;
"""

run_query("RPM/Fuel variance:", physics_cte)


# ================================================================
# TASK 21 — Join CTEs
# ================================================================

print("\n" + "=" * 70)
print("TASK 21 — Static AND physics violation")
print("=" * 70)

static_physics_query = """
WITH Static_Activity AS (
    SELECT *
    FROM Telemetry_Fact_Table
    WHERE CAST(Is_Static_GPS AS INTEGER) = 1
),
Physics_Variance AS (
    SELECT
        Log_ID,
        CAST(Engine_RPM AS REAL) AS RPM,
        CAST(Fuel_LPH AS REAL) AS Fuel
    FROM Telemetry_Fact_Table
)
SELECT
    s.Log_ID,
    s.Vendor,
    s.Equipment_Type,
    s.Billed_Amount_USD,
    p.RPM,
    p.Fuel
FROM Static_Activity s
JOIN Physics_Variance p
    ON s.Log_ID = p.Log_ID
WHERE p.RPM > 1800
AND p.Fuel < 1.0
LIMIT 20;
"""

run_query("Static + physics violations:", static_physics_query)


# ================================================================
# TASK 22 — SELF JOIN duplicate timestamp
# ================================================================

print("\n" + "=" * 70)
print("TASK 22 — SELF JOIN duplicate telemetry")
print("=" * 70)

self_join_query = """
SELECT
    a.Log_ID AS Log_ID_1,
    b.Log_ID AS Log_ID_2,
    a.Vendor,
    a.Telemetry_Date
FROM Telemetry_Fact_Table a
JOIN Telemetry_Fact_Table b
    ON a.Vendor = b.Vendor
    AND a.Telemetry_Date = b.Telemetry_Date
    AND a.Log_ID < b.Log_ID
LIMIT 20;
"""

run_query("Duplicate timestamp records:", self_join_query)


# ================================================================
# TASK 23 — Overlapping timestamp explanation
# ================================================================

print("\n" + "=" * 70)
print("TASK 23 — Overlapping timestamp ranges")
print("=" * 70)

print("""
The dataset does not contain a separate end timestamp for each
billing interval.

Therefore, an exact timestamp-range overlap cannot be calculated
as a true interval-overlap problem.

As a practical audit proxy, consecutive billing records from
the same vendor are compared using LAG().
""")


# ================================================================
# TASK 24 — Implement overlapping timestamp proxy
# ================================================================

print("\n" + "=" * 70)
print("TASK 24 — Overlapping billing audit")
print("=" * 70)

overlap_query = """
WITH VendorTimes AS (
    SELECT
        Log_ID,
        Vendor,
        Telemetry_Date,
        Billed_Engine_Hours,
        LAG(Telemetry_Date)
            OVER (
                PARTITION BY Vendor
                ORDER BY Telemetry_Date, Log_ID
            ) AS Previous_Timestamp,
        LAG(Billed_Engine_Hours)
            OVER (
                PARTITION BY Vendor
                ORDER BY Telemetry_Date, Log_ID
            ) AS Previous_Billed_Hours
    FROM Telemetry_Fact_Table
)
SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    Previous_Timestamp,
    Billed_Engine_Hours,
    Previous_Billed_Hours,
    ROUND(
        (julianday(Telemetry_Date)
        - julianday(Previous_Timestamp)) * 24,
        2
    ) AS Hours_Between_Pings
FROM VendorTimes
WHERE Previous_Timestamp IS NOT NULL
AND CAST(Billed_Engine_Hours AS REAL)
    >
    ROUND(
        (julianday(Telemetry_Date)
        - julianday(Previous_Timestamp)) * 24,
        2
    )
LIMIT 20;
"""

run_query("Potential overlapping/double-billing records:", overlap_query)


# ================================================================
# TASK 25 — HAVING ghost billing > $1M
# ================================================================

print("\n" + "=" * 70)
print("TASK 25 — Vendors with ghost billing > $1 Million")
print("=" * 70)

ghost_having_query = """
SELECT
    Vendor,
    SUM(CAST(Billed_Amount_USD AS REAL)) AS Ghost_Billing_USD
FROM Telemetry_Fact_Table
WHERE CAST(Is_Static_GPS AS INTEGER) = 1
AND CAST(Billed_Engine_Hours AS REAL) > 20
GROUP BY Vendor
HAVING Ghost_Billing_USD > 1000000
ORDER BY Ghost_Billing_USD DESC;
"""

run_query("Ghost billing above $1M:", ghost_having_query)


# ================================================================
# TASK 26 — CAST Engine Hours
# ================================================================

print("\n" + "=" * 70)
print("TASK 26 — CAST Billed_Engine_Hours")
print("=" * 70)

cast_query = """
SELECT
    Log_ID,
    Billed_Engine_Hours,
    CAST(Billed_Engine_Hours AS INTEGER) AS Engine_Hours_Floor
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Floor-based engine hours:", cast_query)


# ================================================================
# TASK 27 — STRFTIME hour
# ================================================================

print("\n" + "=" * 70)
print("TASK 27 — Extract Hour using STRFTIME")
print("=" * 70)

hour_query = """
SELECT
    Log_ID,
    Telemetry_Date,
    STRFTIME('%H', Telemetry_Date) AS Hour
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Extracted hour:", hour_query)


# ================================================================
# TASK 28 — Fraud by hour
# ================================================================

print("\n" + "=" * 70)
print("TASK 28 — Fraud distribution by hour")
print("=" * 70)

fraud_hour_query = """
SELECT
    STRFTIME('%H', Telemetry_Date) AS Hour,
    COUNT(*) AS Total_Logs,
    SUM(
        CASE
            WHEN CAST(Is_Static_GPS AS INTEGER) = 1
            AND CAST(Billed_Engine_Hours AS REAL) > 20
            THEN 1
            ELSE 0
        END
    ) AS Fraud_Logs
FROM Telemetry_Fact_Table
GROUP BY Hour
ORDER BY Hour;
"""

run_query("Fraud by hour:", fraud_hour_query, limit=24)


# ================================================================
# TASK 29 — Fraud percentage by hour
# ================================================================

print("\n" + "=" * 70)
print("TASK 29 — Fraud percentage per hour")
print("=" * 70)

fraud_percentage_query = """
SELECT
    STRFTIME('%H', Telemetry_Date) AS Hour,
    COUNT(*) AS Total_Logs,
    SUM(
        CASE
            WHEN CAST(Is_Static_GPS AS INTEGER) = 1
            AND CAST(Billed_Engine_Hours AS REAL) > 20
            THEN 1
            ELSE 0
        END
    ) AS Fraud_Logs,
    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN CAST(Is_Static_GPS AS INTEGER) = 1
                AND CAST(Billed_Engine_Hours AS REAL) > 20
                THEN 1
                ELSE 0
            END
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS Fraud_Percentage
FROM Telemetry_Fact_Table
GROUP BY Hour
ORDER BY Hour;
"""

run_query("Fraud percentage by hour:", fraud_percentage_query, limit=24)


# ================================================================
# TASK 30 — Quarantine table
# ================================================================

print("\n" + "=" * 70)
print("TASK 30 — Create Quarantine_Logs")
print("=" * 70)

conn.execute("""
CREATE TABLE Quarantine_Logs AS
SELECT *
FROM Telemetry_Fact_Table
WHERE 0;
""")

conn.execute("""
INSERT INTO Quarantine_Logs
SELECT *
FROM Telemetry_Fact_Table
WHERE Vendor = 'Phantom_Leasing'
AND CAST(Is_Static_GPS AS INTEGER) = 1
AND CAST(Billed_Engine_Hours AS REAL) > 20;
""")

conn.commit()

quarantine_count = conn.execute("SELECT COUNT(*) FROM Quarantine_Logs").fetchone()[0]

print(f"Rows moved into Quarantine_Logs: {quarantine_count:,}")


# ================================================================
# TASK 31 — DELETE quarantined rows
# ================================================================

print("\n" + "=" * 70)
print("TASK 31 — Remove quarantined rows from Fact table")
print("=" * 70)

before_delete = conn.execute("SELECT COUNT(*) FROM Telemetry_Fact_Table").fetchone()[0]

conn.execute("""
DELETE FROM Telemetry_Fact_Table
WHERE Vendor = 'Phantom_Leasing'
AND CAST(Is_Static_GPS AS INTEGER) = 1
AND CAST(Billed_Engine_Hours AS REAL) > 20;
""")

conn.commit()

after_delete = conn.execute("SELECT COUNT(*) FROM Telemetry_Fact_Table").fetchone()[0]

deleted_rows = before_delete - after_delete

print(f"Rows before deletion: {before_delete:,}")
print(f"Rows deleted: {deleted_rows:,}")
print(f"Rows remaining: {after_delete:,}")


# ================================================================
# TASK 32 — Negative Fuel trigger
# ================================================================

print("\n" + "=" * 70)
print("TASK 32 — Create negative Fuel_LPH trigger")
print("=" * 70)

conn.execute("""
CREATE TRIGGER prevent_negative_fuel
BEFORE INSERT ON Telemetry_Fact_Table
FOR EACH ROW
WHEN CAST(NEW.Fuel_LPH AS REAL) < 0
BEGIN
    SELECT RAISE(
        ABORT,
        'Negative Fuel_LPH is not allowed'
    );
END;
""")

conn.commit()

print("Trigger created: prevent_negative_fuel")


# ================================================================
# TASK 33 — Test trigger
# ================================================================

print("\n" + "=" * 70)
print("TASK 33 — Test negative fuel trigger")
print("=" * 70)

try:

    conn.execute("""
    INSERT INTO Telemetry_Fact_Table
    (Log_ID, Fuel_LPH, Vendor)
    VALUES
    ('TEST-NEGATIVE-FUEL', '-10', 'TEST_VENDOR');
    """)

    conn.commit()

    print("WARNING: Trigger did not block the insert.")

    conn.execute("""
    DELETE FROM Telemetry_Fact_Table
    WHERE Log_ID = 'TEST-NEGATIVE-FUEL';
    """)

    conn.commit()

except sqlite3.IntegrityError as e:

    print("Trigger successfully blocked negative fuel.")
    print(f"SQLite message: {e}")

except sqlite3.DatabaseError as e:

    print("Trigger successfully blocked negative fuel.")
    print(f"SQLite message: {e}")


# ================================================================
# TASK 34 — COALESCE
# ================================================================

print("\n" + "=" * 70)
print("TASK 34 — COALESCE Is_Static_GPS")
print("=" * 70)

coalesce_query = """
SELECT
    Log_ID,
    COALESCE(Is_Static_GPS, '0') AS Static_GPS_Safe
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("COALESCE result:", coalesce_query)


# ================================================================
# TASK 35 — SUBSTR Site Code
# ================================================================

print("\n" + "=" * 70)
print("TASK 35 — SUBSTR Site prefix")
print("=" * 70)

site_prefix_query = """
SELECT
    Site_Code,
    SUBSTR(Site_Code, 1, 8) AS Site_Prefix
FROM Telemetry_Fact_Table
LIMIT 20;
"""

run_query("Site prefixes:", site_prefix_query)


# ================================================================
# TASK 36 — Fraud distribution by site prefix
# ================================================================

print("\n" + "=" * 70)
print("TASK 36 — Fraud by construction type")
print("=" * 70)

site_fraud_query = """
SELECT
    SUBSTR(Site_Code, 1, 8) AS Site_Prefix,
    COUNT(*) AS Total_Logs,
    SUM(
        CASE
            WHEN CAST(Is_Static_GPS AS INTEGER) = 1
            AND CAST(Billed_Engine_Hours AS REAL) > 20
            THEN 1
            ELSE 0
        END
    ) AS Fraud_Logs,
    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN CAST(Is_Static_GPS AS INTEGER) = 1
                AND CAST(Billed_Engine_Hours AS REAL) > 20
                THEN 1
                ELSE 0
            END
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS Fraud_Percentage
FROM Telemetry_Fact_Table
GROUP BY Site_Prefix
ORDER BY Fraud_Percentage DESC;
"""

run_query("Fraud by site prefix:", site_fraud_query)


# ================================================================
# TASK 37 — FULL OUTER JOIN equivalent
# ================================================================

print("\n" + "=" * 70)
print("TASK 37 — FULL OUTER JOIN equivalent")
print("=" * 70)

full_outer_query = """
SELECT
    v.Vendor,
    e.Equipment_Type
FROM Dim_Vendor v
LEFT JOIN Dim_Equipment e
    ON 1 = 1

UNION ALL

SELECT
    v.Vendor,
    e.Equipment_Type
FROM Dim_Equipment e
LEFT JOIN Dim_Vendor v
    ON 1 = 1
WHERE v.Vendor IS NULL;
"""

run_query("Vendor × Equipment FULL OUTER JOIN equivalent:", full_outer_query)


# ================================================================
# TASK 38 — 5% random sample
# ================================================================

print("\n" + "=" * 70)
print("TASK 38 — 5% manual audit sample")
print("=" * 70)

# SQLite's '%' operator is the practical modulo operator.
sample_query = """
SELECT *
FROM Telemetry_Fact_Table
WHERE ABS(
    CAST(
        SUBSTR(
            REPLACE(Log_ID, 'CRANE-LOG-', ''),
            -8
        ) AS INTEGER
    )
) % 20 = 0
LIMIT 50000;
"""

sample_rows = run_query("5% audit sample:", sample_query, limit=10)

sample_count = conn.execute("""
    SELECT COUNT(*)
    FROM Telemetry_Fact_Table
    WHERE ABS(
        CAST(
            SUBSTR(
                REPLACE(Log_ID, 'CRANE-LOG-', ''),
                -8
            ) AS INTEGER
        )
    ) % 20 = 0;
    """).fetchone()[0]

print(f"Approximate 5% sample rows: {sample_count:,}")


# ================================================================
# TASK 39 — Executive summary VIEW
# ================================================================

print("\n" + "=" * 70)
print("TASK 39 — vw_executive_summary")
print("=" * 70)

conn.execute("DROP VIEW IF EXISTS vw_executive_summary;")

conn.execute("""
CREATE VIEW vw_executive_summary AS
SELECT
    (
        SELECT SUM(CAST(Billed_Amount_USD AS REAL))
        FROM Telemetry_Fact_Table
    )
    +
    (
        SELECT SUM(CAST(Billed_Amount_USD AS REAL))
        FROM Quarantine_Logs
    )
    AS Total_Spend_USD,

    (
        SELECT SUM(CAST(Billed_Amount_USD AS REAL))
        FROM Quarantine_Logs
    )
    AS Total_Ghost_Billing_USD,

    (
        SELECT COUNT(*)
        FROM Quarantine_Logs
    )
    AS Ghost_Log_Count;
""")

conn.commit()

run_query(
    "Executive summary view:",
    """
    SELECT *
    FROM vw_executive_summary;
    """,
)


# ================================================================
# TASK 40 — Physics violations VIEW
# ================================================================

print("\n" + "=" * 70)
print("TASK 40 — vw_physics_violations")
print("=" * 70)

conn.execute("DROP VIEW IF EXISTS vw_physics_violations;")

conn.execute("""
CREATE VIEW vw_physics_violations AS
SELECT
    Log_ID,
    Vendor,
    Equipment_Type,
    Telemetry_Date,
    Engine_RPM,
    Fuel_LPH,
    ROUND(
        CAST(Engine_RPM AS REAL)
        -
        CAST(Fuel_LPH AS REAL),
        4
    ) AS RPM_Fuel_Discrepancy
FROM Telemetry_Fact_Table
WHERE CAST(Engine_RPM AS REAL) > 1800
AND CAST(Fuel_LPH AS REAL) < 1.0;
""")

conn.commit()

run_query(
    "Physics violations:",
    """
    SELECT *
    FROM vw_physics_violations
    LIMIT 20;
    """,
)


# ================================================================
# TASK 41 — Export executive summary
# ================================================================

print("\n" + "=" * 70)
print("TASK 41 — Export executive summary CSV")
print("=" * 70)

export_query_to_csv(
    """
    SELECT *
    FROM vw_executive_summary;
    """,
    EXECUTIVE_CSV,
)


# ================================================================
# TASK 42 — Export physics violations
# ================================================================

print("\n" + "=" * 70)
print("TASK 42 — Export physics violations CSV")
print("=" * 70)

export_query_to_csv(
    """
    SELECT *
    FROM vw_physics_violations;
    """,
    PHYSICS_CSV,
)


# ================================================================
# TASK 43 — DISTINCT Latitude
# ================================================================

print("\n" + "=" * 70)
print("TASK 43 — Phantom_Leasing distinct locations")
print("=" * 70)

distinct_location_query = """
SELECT
    COUNT(DISTINCT Latitude) AS Distinct_Latitudes,
    COUNT(DISTINCT Longitude) AS Distinct_Longitudes
FROM Quarantine_Logs;
"""

run_query("Phantom_Leasing quarantined GPS location count:", distinct_location_query)


# ================================================================
# TASK 44 — Vendor fraud concentration ratio
# ================================================================

print("\n" + "=" * 70)
print("TASK 44 — Fraud concentration ratio")
print("=" * 70)

concentration_query = """
SELECT
    Vendor,
    Vendor_Ghost_Billing,
    ROUND(
        100.0 * Vendor_Ghost_Billing
        /
        NULLIF(
            (
                SELECT SUM(CAST(Billed_Amount_USD AS REAL))
                FROM Quarantine_Logs
            ),
            0
        ),
        2
    ) AS Fraud_Concentration_Percentage
FROM (
    SELECT
        Vendor,
        SUM(CAST(Billed_Amount_USD AS REAL))
            AS Vendor_Ghost_Billing
    FROM Quarantine_Logs
    GROUP BY Vendor
)
ORDER BY Vendor_Ghost_Billing DESC;
"""

run_query("Fraud concentration by vendor:", concentration_query)


# ================================================================
# TASK 45 — Lowest fraud percentage site
# ================================================================

print("\n" + "=" * 70)
print("TASK 45 — Lowest fraud percentage site")
print("=" * 70)

safest_site_query = """
SELECT
    s.Site_Code,
    (
        SELECT COUNT(*)
        FROM Telemetry_Fact_Table f
        WHERE f.Site_Code = s.Site_Code
    ) AS Total_Logs,

    (
        SELECT COUNT(*)
        FROM Quarantine_Logs q
        WHERE q.Site_Code = s.Site_Code
    ) AS Fraud_Logs,

    ROUND(
        100.0 *
        (
            SELECT COUNT(*)
            FROM Quarantine_Logs q
            WHERE q.Site_Code = s.Site_Code
        )
        /
        NULLIF(
            (
                SELECT COUNT(*)
                FROM Telemetry_Fact_Table f
                WHERE f.Site_Code = s.Site_Code
            )
            +
            (
                SELECT COUNT(*)
                FROM Quarantine_Logs q2
                WHERE q2.Site_Code = s.Site_Code
            ),
            0
        ),
        2
    ) AS Fraud_Percentage

FROM Dim_Site s
ORDER BY Fraud_Percentage ASC
LIMIT 1;
"""

run_query("Site with lowest fraud percentage:", safest_site_query)


# ================================================================
# TASK 46 — EXISTS
# ================================================================

print("\n" + "=" * 70)
print("TASK 46 — EXISTS moving GPS check")
print("=" * 70)

exists_query = """
SELECT
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM Telemetry_Fact_Table
            WHERE Vendor = 'Phantom_Leasing'
            AND CAST(Is_Static_GPS AS INTEGER) = 0
        )
        OR EXISTS (
            SELECT 1
            FROM Quarantine_Logs
            WHERE Vendor = 'Phantom_Leasing'
            AND CAST(Is_Static_GPS AS INTEGER) = 0
        )
        THEN 'YES — moving GPS exists'
        ELSE 'NO — no moving GPS found'
    END AS Phantom_Moving_GPS_Result;
"""

run_query("Phantom_Leasing moving GPS test:", exists_query)


# ================================================================
# TASK 47 — NOT IN trusted vendors
# ================================================================

print("\n" + "=" * 70)
print("TASK 47 — NOT IN trusted vendors")
print("=" * 70)

trusted_vendors = ("Apex_Rentals", "Global_Equip", "Titan_Heavy_Machinery")

not_in_query = f"""
SELECT
    Vendor,
    COUNT(*) AS Audit_Records,
    SUM(CAST(Billed_Amount_USD AS REAL))
        AS Audit_Billing_USD
FROM Telemetry_Fact_Table
WHERE Vendor NOT IN ({",".join(["?"] * len(trusted_vendors))})
GROUP BY Vendor;
"""

run_query("Vendors outside trusted-vendor list:", not_in_query, params=trusted_vendors)


# ================================================================
# TASK 48 — Complex CTE execution time
# ================================================================

print("\n" + "=" * 70)
print("TASK 48 — Complex CTE execution time")
print("=" * 70)

complex_cte = """
WITH Static_Activity AS (
    SELECT
        Log_ID,
        Vendor,
        Equipment_Type,
        Site_Code,
        Telemetry_Date,
        Billed_Engine_Hours,
        Billed_Amount_USD,
        Engine_RPM,
        Fuel_LPH,
        Is_Static_GPS
    FROM Quarantine_Logs
    WHERE CAST(Is_Static_GPS AS INTEGER) = 1
),
VendorRunning AS (
    SELECT
        Log_ID,
        Vendor,
        Telemetry_Date,
        Billed_Amount_USD,
        SUM(CAST(Billed_Amount_USD AS REAL))
            OVER (
                PARTITION BY Vendor
                ORDER BY Telemetry_Date, Log_ID
            ) AS Running_Billing
    FROM Static_Activity
),
PhysicsCheck AS (
    SELECT
        Log_ID,
        Engine_RPM,
        Fuel_LPH,
        CAST(Engine_RPM AS REAL)
        -
        CAST(Fuel_LPH AS REAL)
        AS RPM_Fuel_Variance
    FROM Quarantine_Logs
)
SELECT
    v.Vendor,
    COUNT(*) AS Records,
    SUM(CAST(v.Billed_Amount_USD AS REAL))
        AS Total_Billing,
    MAX(p.RPM_Fuel_Variance)
        AS Max_RPM_Fuel_Variance
FROM VendorRunning v
JOIN PhysicsCheck p
    ON v.Log_ID = p.Log_ID
GROUP BY v.Vendor
ORDER BY Total_Billing DESC;
"""

start_time = time.perf_counter()

cur = conn.cursor()
cur.execute(complex_cte)
complex_results = cur.fetchall()

end_time = time.perf_counter()

execution_ms = (end_time - start_time) * 1000

for row in complex_results:
    print(row)

print(f"\nComplex CTE execution time: {execution_ms:.3f} ms")


# ================================================================
# TASK 49 — VACUUM
# ================================================================

print("\n" + "=" * 70)
print("TASK 49 — VACUUM database")
print("=" * 70)

conn.commit()

before_vacuum_size = DB_FILE.stat().st_size

print(f"Database size before VACUUM: " f"{before_vacuum_size / (1024 * 1024):.2f} MB")

conn.execute("VACUUM;")

after_vacuum_size = DB_FILE.stat().st_size

print(f"Database size after VACUUM: " f"{after_vacuum_size / (1024 * 1024):.2f} MB")


# ================================================================
# TASK 50 — Close database
# ================================================================

print("\n" + "=" * 70)
print("TASK 50 — Close database connection")
print("=" * 70)

conn.commit()
conn.close()

print("SQLite database connection closed.")
print("File locks released.")


# ================================================================
# TASK 51 — Package SQL scripts
# ================================================================

print("\n" + "=" * 70)
print("TASK 51 — Create telematics_audit.sql")
print("=" * 70)

sql_package = r"""
-- ================================================================
-- NEXLYRA PROJECT 4
-- DAY 4 — ADVANCED SQL & RELATIONAL DATA MODELING
-- Carol
-- ================================================================

-- DATABASE
-- nexlyra_telematics.db

-- ================================================================
-- DIMENSION TABLES
-- ================================================================

CREATE TABLE IF NOT EXISTS Dim_Vendor (
    Vendor_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Vendor TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Equipment (
    Equipment_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Equipment_Type TEXT UNIQUE NOT NULL,
    Standard_Rental_Rate_USD REAL
);

CREATE TABLE IF NOT EXISTS Dim_Site (
    Site_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Site_Code TEXT UNIQUE NOT NULL,
    Project_Zone TEXT
);

-- ================================================================
-- INDEXES
-- ================================================================

CREATE INDEX IF NOT EXISTS idx_fact_vendor
ON Telemetry_Fact_Table(Vendor);

CREATE INDEX IF NOT EXISTS idx_fact_telemetry_date
ON Telemetry_Fact_Table(Telemetry_Date);

-- ================================================================
-- TASK 11
-- TOTAL BILLING PER VENDOR
-- ================================================================

SELECT
    Vendor,
    SUM(CAST(Billed_Amount_USD AS REAL)) AS Total_Billed_USD
FROM Telemetry_Fact_Table
GROUP BY Vendor
ORDER BY Total_Billed_USD DESC;

-- ================================================================
-- TASK 12
-- AVERAGE FUEL PER EQUIPMENT
-- ================================================================

SELECT
    Equipment_Type,
    AVG(CAST(Fuel_LPH AS REAL)) AS Average_Fuel_LPH
FROM Telemetry_Fact_Table
GROUP BY Equipment_Type;

-- ================================================================
-- TASK 13
-- FUEL SUSPICION
-- ================================================================

SELECT
    Log_ID,
    Vendor,
    Fuel_LPH,
    CASE
        WHEN CAST(Fuel_LPH AS REAL) < 0.5
        THEN 'High_Suspicion'
        ELSE 'Normal'
    END AS Suspicion_Category
FROM Telemetry_Fact_Table;

-- ================================================================
-- TASK 14-15
-- WINDOW FUNCTION / RUNNING TOTAL
-- ================================================================

SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    Billed_Amount_USD,
    SUM(CAST(Billed_Amount_USD AS REAL))
        OVER (
            PARTITION BY Vendor
            ORDER BY Telemetry_Date, Log_ID
        ) AS Running_Total_USD
FROM Telemetry_Fact_Table;

-- ================================================================
-- TASK 16
-- LAG
-- ================================================================

SELECT
    Log_ID,
    Vendor,
    Telemetry_Date,
    LAG(Telemetry_Date)
        OVER (
            PARTITION BY Vendor
            ORDER BY Telemetry_Date, Log_ID
        ) AS Previous_Timestamp
FROM Telemetry_Fact_Table
WHERE Vendor = 'Phantom_Leasing';

-- ================================================================
-- TASK 17
-- LEAD
-- ================================================================

SELECT
    Log_ID,
    Engine_RPM,
    LEAD(Engine_RPM)
        OVER (
            PARTITION BY Vendor
            ORDER BY Telemetry_Date, Log_ID
        ) AS Next_Engine_RPM
FROM Telemetry_Fact_Table;

-- ================================================================
-- TASK 18-21
-- STATIC / PHYSICS CTE
-- ================================================================

WITH Static_Activity AS (
    SELECT *
    FROM Telemetry_Fact_Table
    WHERE CAST(Is_Static_GPS AS INTEGER) = 1
),
Physics_Variance AS (
    SELECT
        Log_ID,
        CAST(Engine_RPM AS REAL) AS RPM,
        CAST(Fuel_LPH AS REAL) AS Fuel
    FROM Telemetry_Fact_Table
)
SELECT
    s.Log_ID,
    s.Vendor,
    p.RPM,
    p.Fuel
FROM Static_Activity s
JOIN Physics_Variance p
    ON s.Log_ID = p.Log_ID
WHERE p.RPM > 1800
AND p.Fuel < 1.0;

-- ================================================================
-- TASK 22
-- SELF JOIN
-- ================================================================

SELECT
    a.Log_ID AS Log_ID_1,
    b.Log_ID AS Log_ID_2,
    a.Vendor,
    a.Telemetry_Date
FROM Telemetry_Fact_Table a
JOIN Telemetry_Fact_Table b
    ON a.Vendor = b.Vendor
    AND a.Telemetry_Date = b.Telemetry_Date
    AND a.Log_ID < b.Log_ID;

-- ================================================================
-- TASK 25
-- HAVING
-- ================================================================

SELECT
    Vendor,
    SUM(CAST(Billed_Amount_USD AS REAL))
        AS Ghost_Billing_USD
FROM Telemetry_Fact_Table
WHERE CAST(Is_Static_GPS AS INTEGER) = 1
AND CAST(Billed_Engine_Hours AS REAL) > 20
GROUP BY Vendor
HAVING Ghost_Billing_USD > 1000000;

-- ================================================================
-- TASK 27-29
-- HOURLY FRAUD
-- ================================================================

SELECT
    STRFTIME('%H', Telemetry_Date) AS Hour,
    COUNT(*) AS Total_Logs,
    SUM(
        CASE
            WHEN CAST(Is_Static_GPS AS INTEGER) = 1
            AND CAST(Billed_Engine_Hours AS REAL) > 20
            THEN 1
            ELSE 0
        END
    ) AS Fraud_Logs
FROM Telemetry_Fact_Table
GROUP BY Hour;

-- ================================================================
-- TASK 32
-- NEGATIVE FUEL TRIGGER
-- ================================================================

CREATE TRIGGER IF NOT EXISTS prevent_negative_fuel
BEFORE INSERT ON Telemetry_Fact_Table
FOR EACH ROW
WHEN CAST(NEW.Fuel_LPH AS REAL) < 0
BEGIN
    SELECT RAISE(
        ABORT,
        'Negative Fuel_LPH is not allowed'
    );
END;

-- ================================================================
-- TASK 34
-- COALESCE
-- ================================================================

SELECT
    Log_ID,
    COALESCE(Is_Static_GPS, '0') AS Static_GPS_Safe
FROM Telemetry_Fact_Table;

-- ================================================================
-- TASK 35
-- SUBSTR
-- ================================================================

SELECT
    Site_Code,
    SUBSTR(Site_Code, 1, 8) AS Site_Prefix
FROM Telemetry_Fact_Table;

-- ================================================================
-- TASK 37
-- FULL OUTER JOIN EQUIVALENT
-- ================================================================

SELECT
    v.Vendor,
    e.Equipment_Type
FROM Dim_Vendor v
LEFT JOIN Dim_Equipment e
    ON 1 = 1

UNION ALL

SELECT
    v.Vendor,
    e.Equipment_Type
FROM Dim_Equipment e
LEFT JOIN Dim_Vendor v
    ON 1 = 1
WHERE v.Vendor IS NULL;

-- ================================================================
-- TASK 39
-- EXECUTIVE SUMMARY VIEW
-- ================================================================

CREATE VIEW IF NOT EXISTS vw_executive_summary AS
SELECT
    (
        SELECT SUM(CAST(Billed_Amount_USD AS REAL))
        FROM Telemetry_Fact_Table
    ) AS Total_Spend_USD,
    (
        SELECT SUM(CAST(Billed_Amount_USD AS REAL))
        FROM Quarantine_Logs
    ) AS Total_Ghost_Billing_USD;

-- ================================================================
-- TASK 40
-- PHYSICS VIOLATIONS VIEW
-- ================================================================

CREATE VIEW IF NOT EXISTS vw_physics_violations AS
SELECT
    Log_ID,
    Vendor,
    Equipment_Type,
    Engine_RPM,
    Fuel_LPH,
    CAST(Engine_RPM AS REAL)
    -
    CAST(Fuel_LPH AS REAL)
    AS RPM_Fuel_Discrepancy
FROM Telemetry_Fact_Table
WHERE CAST(Engine_RPM AS REAL) > 1800
AND CAST(Fuel_LPH AS REAL) < 1.0;

-- ================================================================
-- END OF TELEMATICS AUDIT SQL PACKAGE
-- ================================================================
"""

with open(AUDIT_SQL_FILE, "w", encoding="utf-8") as f:
    f.write(sql_package)

print(f"SQL package created: {AUDIT_SQL_FILE.name}")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("DAY 4 — FINAL SUMMARY")
print("=" * 70)

print(f"Database: {DB_FILE.name}")
print(f"Master CSV: {CSV_FILE.name}")
print(f"Rows originally imported: {row_count:,}")
print(f"Rows quarantined: {quarantine_count:,}")
print(f"Rows deleted from Fact table: {deleted_rows:,}")

print("\nOutput files:")

for file in [DB_FILE, EXECUTIVE_CSV, PHYSICS_CSV, AUDIT_SQL_FILE]:
    print(f"{file.name} → {file.exists()}")

print("\n" + "=" * 70)
print("DAY 4 PROCESSING FINISHED")
print("=" * 70)
