# Nexlyra Crane Telematics Fraud Audit

## Executive summary
This project audits one million crane equipment-hour billing records across four vendors. It detects records where equipment reports a static GPS position while continuing to accrue billed engine hours.

The Ghost Equipment rule is:

```text
Vendor = Phantom_Leasing
AND Is_Static_GPS = TRUE
AND Billed_Engine_Hours > 20
```

It identifies **83,281 records** and **$692,257,429.65 in identified ghost billing**, against **$2,829,521,843.48 in total billed spend**. This is a testable billing pattern requiring physical inspection and vendor verification, not proof of confirmed fraud.

## Verified findings

| Metric | Result |
|---|---:|
| Dataset size | 1,000,000 rows |
| Total billed | $2,829,521,843.48 |
| Identified Ghost Equipment logs | 83,281 |
| Identified Ghost Billing | $692,257,429.65 |
| Static GPS records across all vendors | 83,371 |
| Exact deterministic damages condition | $0 |

Flagged Phantom_Leasing records share one coordinate pair. Moving Phantom_Leasing records also exist, so vendor identity alone was not used as the rule.

## Methodology
- Decode base64 GPS payloads and parse latitude/longitude.
- Compare consecutive readings by equipment type to derive `Is_Static_GPS`.
- Cross-reference static positions, billed hours, vendor, and equipment type.
- Use SQL relational modeling and a Power BI star schema.
- Apply variance, OLS, correlation, Kolmogorov-Smirnov, and Isolation Forest diagnostics.

RPM and fuel fields for the affected Tower Crane subset had effectively zero variance. Undefined OLS/correlation and all-row Isolation Forest flags reflect this data limitation, not independent proof of fraud.

## Controls and limitations
Recommended controls include physical inspection, vendor verification, GPS/device provenance validation, invoice reconciliation, and preservation of raw telemetry and audit outputs. Static GPS can have legitimate explanations; confirmed conclusions require physical inspection, source-system validation, contracts, invoices, payment evidence, and vendor responses.

## Repository contents
Telemetry data, cleaned datasets, SQL audit logic, Python analysis scripts, Power BI dashboards, reports, and visual evidence.
