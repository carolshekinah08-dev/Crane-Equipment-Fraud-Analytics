
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
