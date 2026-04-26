from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

con = connect_mimic_duckdb()


def q(sql: str) -> tuple:
    row = con.execute(sql).fetchone()
    assert row is not None
    return row


# Basic counts
n_patients, n_admissions, n_hosp_patients, n_icu_stays, n_icu_patients, n_radiology = q("""--sql
    SELECT
        (SELECT COUNT(DISTINCT subject_id) FROM mimiciv_hosp.patients)            AS n_patients,
        (SELECT COUNT(*)                   FROM mimiciv_hosp.admissions)           AS n_admissions,
        (SELECT COUNT(DISTINCT subject_id) FROM mimiciv_hosp.admissions)           AS n_hosp_patients,
        (SELECT COUNT(*)                   FROM mimiciv_icu.icustays)              AS n_icu_stays,
        (SELECT COUNT(DISTINCT subject_id) FROM mimiciv_icu.icustays)              AS n_icu_patients,
        (SELECT COUNT(*)                   FROM mimiciv_note.radiology)            AS n_radiology,
""")

n_discharge, n_discharge_patients = q("""--sql
    SELECT COUNT(*), COUNT(DISTINCT subject_id)
    FROM mimiciv_note.discharge
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
hosp_age_mean, hosp_age_std, hosp_age_median, icu_age_mean, icu_age_std, icu_age_median = q("""--sql
    WITH ages AS (
        SELECT
            mimiciv_hosp.admissions.hadm_id,
            mimiciv_hosp.patients.anchor_age + (YEAR(mimiciv_hosp.admissions.admittime::TIMESTAMP) - mimiciv_hosp.patients.anchor_year) AS age_at_admit,
            EXISTS (
                SELECT 1 FROM mimiciv_icu.icustays WHERE mimiciv_icu.icustays.hadm_id = mimiciv_hosp.admissions.hadm_id
            ) AS is_icu
        FROM mimiciv_hosp.admissions
        JOIN mimiciv_hosp.patients USING (subject_id)
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
n_female_hosp, n_hosp_total, n_female_icu, n_icu_total = q("""--sql
    SELECT
        COUNTIF(mimiciv_hosp.patients.gender = 'F')                                                  AS n_female_hosp,
        COUNT(*)                                                                                      AS n_hosp_total,
        COUNTIF(mimiciv_hosp.patients.gender = 'F' AND mimiciv_icu.icustays.hadm_id IS NOT NULL)      AS n_female_icu,
        COUNT(mimiciv_icu.icustays.hadm_id)                                                           AS n_icu_total,
    FROM mimiciv_hosp.admissions
    JOIN mimiciv_hosp.patients USING (subject_id)
    LEFT JOIN (SELECT DISTINCT hadm_id FROM mimiciv_icu.icustays) AS mimiciv_icu.icustays
        ON mimiciv_hosp.admissions.hadm_id = mimiciv_icu.icustays.hadm_id
""")
pct_female_hosp = n_female_hosp / n_hosp_total * 100
pct_female_icu = n_female_icu / n_icu_total * 100

print(f'\nFemale admissions (hospital): {n_female_hosp:,} ({pct_female_hosp:.1f}%)')
print(f'Female admissions (ICU):      {n_female_icu:,} ({pct_female_icu:.1f}%)')

# Length of stay
hosp_los = q("""--sql
    SELECT
        AVG(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
        STDDEV_SAMP(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
        MEDIAN(epoch(dischtime::TIMESTAMP - admittime::TIMESTAMP) / 86400.0),
    FROM mimiciv_hosp.admissions
""")

icu_los = q("""
    SELECT AVG(los), STDDEV_SAMP(los), MEDIAN(los)
    FROM mimiciv_icu.icustays
""")

print(f'\nLOS hospital (days):  {hosp_los[0]:.1f} ± {hosp_los[1]:.1f}  [median {hosp_los[2]:.1f}]')
print(f'LOS ICU (days):       {icu_los[0]:.1f} ± {icu_los[1]:.1f}  [median {icu_los[2]:.1f}]')

# Mortality
n_hosp_death, n_hosp_total2, n_icu_death, n_icu_total2 = q("""--sql
    SELECT
        SUM(mimiciv_hosp.admissions.hospital_expire_flag)                                                        AS n_hosp_death,
        COUNT(*)                                                                                                  AS n_hosp_total,
        SUM(mimiciv_hosp.admissions.hospital_expire_flag) FILTER (WHERE icu_distinct.hadm_id IS NOT NULL)         AS n_icu_death,
        COUNT(icu_distinct.hadm_id)                                                                               AS n_icu_total,
    FROM mimiciv_hosp.admissions
    LEFT JOIN (SELECT DISTINCT hadm_id FROM mimiciv_icu.icustays) AS icu_distinct
        ON mimiciv_hosp.admissions.hadm_id = icu_distinct.hadm_id
""")
pct_hosp_death = n_hosp_death / n_hosp_total2 * 100
pct_icu_death = n_icu_death / n_icu_total2 * 100

print(f'\nIn-hospital mortality:  {n_hosp_death:,} ({pct_hosp_death:.1f}%)')
print(f'ICU mortality:          {n_icu_death:,} ({pct_icu_death:.1f}%)')

# One-year mortality
# dod from patients; one-year = died within 365 days of first admission
n_1y_hosp_death, n_1y_hosp_total, n_1y_icu_death, n_1y_icu_total = q("""--sql
    WITH first_admit AS (
        SELECT subject_id, MIN(admittime::TIMESTAMP) AS first_admittime
        FROM mimiciv_hosp.admissions
        GROUP BY subject_id
    ),
    cohort AS (
        SELECT
            first_admit.subject_id,
            first_admit.first_admittime,
            mimiciv_hosp.patients.dod::TIMESTAMP AS dod,
            dod - first_admittime <= INTERVAL 365 DAYS AS died_1y,
            EXISTS (
                SELECT 1 FROM mimiciv_icu.icustays
                WHERE mimiciv_icu.icustays.subject_id = first_admit.subject_id
            ) AS had_icu
        FROM first_admit
        JOIN mimiciv_hosp.patients USING (subject_id)
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
