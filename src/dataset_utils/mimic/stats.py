from pathlib import Path

import duckdb

MIMIC_IV = (
    Path(__file__).resolve().parent.parent.parent.parent / 'datasets' / 'full-data' / 'mimic-iv'
)

hosp = str(MIMIC_IV / 'hosp')
icu = str(MIMIC_IV / 'icu')
note = str(MIMIC_IV / 'note')

con = duckdb.connect()


def q(sql: str) -> tuple:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row


# Basic counts
n_patients, n_admissions, n_hosp_patients, n_icu_stays, n_icu_patients, n_radiology = q(f"""
    SELECT
        (SELECT COUNT(DISTINCT subject_id) FROM read_csv_auto('{hosp}/patients.csv'))   AS n_patients,
        (SELECT COUNT(*)                   FROM read_csv_auto('{hosp}/admissions.csv')) AS n_admissions,
        (SELECT COUNT(DISTINCT subject_id) FROM read_csv_auto('{hosp}/admissions.csv')) AS n_hosp_patients,
        (SELECT COUNT(*)                   FROM read_csv_auto('{icu}/icustays.csv'))    AS n_icu_stays,
        (SELECT COUNT(DISTINCT subject_id) FROM read_csv_auto('{icu}/icustays.csv'))    AS n_icu_patients,
        (SELECT COUNT(*)                   FROM read_csv_auto('{note}/radiology.csv'))  AS n_radiology,
""")

n_discharge, n_discharge_patients = q(f"""
    SELECT COUNT(*), COUNT(DISTINCT subject_id)
    FROM read_csv_auto('{note}/discharge.csv')
""")

print('MIMIC-IV v3.1 Statistics')
print(f'{"=" * 50}')
print(f'Unique patients (total):     {n_patients:>10,}')
print(f'  with hospitalizations:     {n_hosp_patients:>10,}')
print(f'  ED-only (no admission):    {n_patients - n_hosp_patients:>10,}')
print(f'Hospital admissions:         {n_admissions:>10,}')
print(f'ICU stays:                   {n_icu_stays:>10,}')
print(f'Unique ICU patients:         {n_icu_patients:>10,}')

# Age at admission
# anchor_age is patient age at anchor_year; age at admit = anchor_age + (admit_year - anchor_year)
hosp_age_mean, hosp_age_std, hosp_age_median, icu_age_mean, icu_age_std, icu_age_median = q(f"""
    WITH ages AS (
        SELECT
            a.hadm_id,
            p.anchor_age + (YEAR(a.admittime::TIMESTAMP) - p.anchor_year) AS age_at_admit,
            EXISTS (
                SELECT 1 FROM read_csv_auto('{icu}/icustays.csv') i WHERE i.hadm_id = a.hadm_id
            ) AS is_icu
        FROM read_csv_auto('{hosp}/admissions.csv') a
        JOIN read_csv_auto('{hosp}/patients.csv')   p USING (subject_id)
    )
    SELECT
        AVG(age_at_admit)                               AS hosp_age_mean,
        STDDEV_SAMP(age_at_admit)                       AS hosp_age_std,
        MEDIAN(age_at_admit)                            AS hosp_age_median,
        AVG(age_at_admit) FILTER (WHERE is_icu)         AS icu_age_mean,
        STDDEV_SAMP(age_at_admit) FILTER (WHERE is_icu) AS icu_age_std,
        MEDIAN(age_at_admit) FILTER (WHERE is_icu)      AS icu_age_median,
    FROM ages
""")

print(
    f'\nAge at admission (hospital): {hosp_age_mean:.1f} ± {hosp_age_std:.1f}  [median {hosp_age_median:.0f}]'
)
print(
    f'Age at admission (ICU):      {icu_age_mean:.1f} ± {icu_age_std:.1f}  [median {icu_age_median:.0f}]'
)

# Gender
n_female_hosp, n_hosp_total, n_female_icu, n_icu_total = q(f"""
    SELECT
        COUNTIF(p.gender = 'F')                             AS n_female_hosp,
        COUNT(*)                                            AS n_hosp_total,
        COUNTIF(p.gender = 'F' AND i.hadm_id IS NOT NULL)   AS n_female_icu,
        COUNT(i.hadm_id)                                    AS n_icu_total,
    FROM read_csv_auto('{hosp}/admissions.csv') a
    JOIN read_csv_auto('{hosp}/patients.csv')   p USING (subject_id)
    LEFT JOIN (SELECT DISTINCT hadm_id FROM read_csv_auto('{icu}/icustays.csv')) i
        ON a.hadm_id = i.hadm_id
""")
pct_female_hosp = n_female_hosp / n_hosp_total * 100
pct_female_icu = n_female_icu / n_icu_total * 100

print(f'\nFemale admissions (hospital): {n_female_hosp:,} ({pct_female_hosp:.1f}%)')
print(f'Female admissions (ICU):      {n_female_icu:,} ({pct_female_icu:.1f}%)')

# Length of stay
hosp_los = q(f"""
    SELECT
        AVG(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
        STDDEV_SAMP(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
        MEDIAN(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
    FROM read_csv_auto('{hosp}/admissions.csv')
""")

icu_los = q(f"""
    SELECT AVG(los), STDDEV_SAMP(los), MEDIAN(los)
    FROM read_csv_auto('{icu}/icustays.csv')
""")

print(f'\nLOS hospital (days):  {hosp_los[0]:.1f} ± {hosp_los[1]:.1f}  [median {hosp_los[2]:.1f}]')
print(f'LOS ICU (days):       {icu_los[0]:.1f} ± {icu_los[1]:.1f}  [median {icu_los[2]:.1f}]')

# Mortality
n_hosp_death, n_hosp_total2, n_icu_death, n_icu_total2 = q(f"""
    SELECT
        SUM(hospital_expire_flag)                                      AS n_hosp_death,
        COUNT(*)                                                       AS n_hosp_total,
        SUM(hospital_expire_flag) FILTER (WHERE i.hadm_id IS NOT NULL) AS n_icu_death,
        COUNT(i.hadm_id)                                               AS n_icu_total,
    FROM read_csv_auto('{hosp}/admissions.csv') a
    LEFT JOIN (SELECT DISTINCT hadm_id FROM read_csv_auto('{icu}/icustays.csv')) i
        ON a.hadm_id = i.hadm_id
""")
pct_hosp_death = n_hosp_death / n_hosp_total2 * 100
pct_icu_death = n_icu_death / n_icu_total2 * 100

print(f'\nIn-hospital mortality:  {n_hosp_death:,} ({pct_hosp_death:.1f}%)')
print(f'ICU mortality:          {n_icu_death:,} ({pct_icu_death:.1f}%)')

# One-year mortality
# dod from patients; one-year = died within 365 days of first admission
n_1y_hosp_death, n_1y_hosp_total, n_1y_icu_death, n_1y_icu_total = q(f"""
    WITH first_admit AS (
        SELECT subject_id, MIN(admittime::TIMESTAMP) AS first_admittime
        FROM read_csv_auto('{hosp}/admissions.csv')
        GROUP BY subject_id
    ),
    cohort AS (
        SELECT
            f.subject_id,
            f.first_admittime,
            p.dod::TIMESTAMP AS dod,
            dod - first_admittime <= INTERVAL 365 DAYS AS died_1y,
            EXISTS (
                SELECT 1 FROM read_csv_auto('{icu}/icustays.csv') i
                WHERE i.subject_id = f.subject_id
            ) AS had_icu
        FROM first_admit f
        JOIN read_csv_auto('{hosp}/patients.csv') p USING (subject_id)
    )
    SELECT
        COUNTIF(died_1y)                    AS n_1y_hosp_death,
        COUNT(*)                            AS n_1y_hosp_total,
        COUNTIF(died_1y AND had_icu)        AS n_1y_icu_death,
        COUNTIF(had_icu)                    AS n_1y_icu_total,
    FROM cohort
""")
pct_1y_hosp = n_1y_hosp_death / n_1y_hosp_total * 100
pct_1y_icu = n_1y_icu_death / n_1y_icu_total * 100

print(f'One year mortality (hospital): {n_1y_hosp_death:,} ({pct_1y_hosp:.1f}%)')
print(f'One year mortality (ICU):      {n_1y_icu_death:,} ({pct_1y_icu:.1f}%)')

# Notes
print(f'\nDischarge summaries: {n_discharge:,} from {n_discharge_patients:,} patients')
print(f'Radiology reports:   {n_radiology:,}')

# Slide table
print(f'\n{"=" * 50}')
print('Slide table values:')
print(f'{"=" * 50}')
print(f'["Stays", "{n_admissions:,}", "{n_icu_stays:,}"],')
print(f'["Unique patients", "{n_hosp_patients:,}", "{n_icu_patients:,}"],')
print(
    f'["Age, mean (SD)", "{hosp_age_mean:.1f} ({hosp_age_std:.1f})", "{icu_age_mean:.1f} ({icu_age_std:.1f})"],'
)
print(
    f'["Female, n (%)", "{n_female_hosp:,} ({pct_female_hosp:.1f})", "{n_female_icu:,} ({pct_female_icu:.1f})"],'
)
print(
    f'["Length of stay, mean (SD)", "{hosp_los[0]:.1f} ({hosp_los[1]:.1f})", "{icu_los[0]:.1f} ({icu_los[1]:.1f})"],'
)
print(
    f'["In-hospital mortality, n (%)", "{n_hosp_death:,} ({pct_hosp_death:.1f})", "{n_icu_death:,} ({pct_icu_death:.1f})"],'
)
print(f'["Discharge summaries", "{n_discharge:,}", "\u2014"],')
print(f'["Radiology reports", "{n_radiology:,}", "\u2014"],')
