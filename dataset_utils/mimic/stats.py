from pathlib import Path

import pandas as pd

MIMIC_IV = Path(__file__).resolve().parent.parent.parent / 'datasets' / 'full-data' / 'mimic-iv'

#  Load tables
patients = pd.read_csv(MIMIC_IV / 'hosp' / 'patients.csv')
admissions = pd.read_csv(MIMIC_IV / 'hosp' / 'admissions.csv')
icustays = pd.read_csv(MIMIC_IV / 'icu' / 'icustays.csv')
discharge = pd.read_csv(
    MIMIC_IV / 'note' / 'discharge.csv', usecols=['note_id', 'subject_id', 'hadm_id']
)
radiology = pd.read_csv(MIMIC_IV / 'note' / 'radiology.csv', usecols=['note_id'])

#  Basic counts
n_patients = patients['subject_id'].nunique()
n_admissions = len(admissions)
n_icu_stays = len(icustays)
n_hosp_patients = admissions['subject_id'].nunique()
n_icu_patients = icustays['subject_id'].nunique()

print('MIMIC-IV v3.1 Statistics')
print(f'{"=" * 50}')
print(f'Unique patients (total):     {n_patients:>10,}')
print(f'  with hospitalizations:     {n_hosp_patients:>10,}')
print(f'  ED-only (no admission):    {n_patients - n_hosp_patients:>10,}')
print(f'Hospital admissions:         {n_admissions:>10,}')
print(f'ICU stays:                   {n_icu_stays:>10,}')
print(f'Unique ICU patients:         {n_icu_patients:>10,}')

#  Age
# anchor_age is age at anchor_year; compute age at admission
admissions = admissions.merge(
    patients[['subject_id', 'anchor_age', 'anchor_year']], on='subject_id', how='left'
)
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['age_at_admit'] = admissions['anchor_age'] + (
    admissions['admittime'].dt.year - admissions['anchor_year']
)

hosp_age = admissions['age_at_admit']
icu_admit = admissions[admissions['hadm_id'].isin(icustays['hadm_id'])]
icu_age = icu_admit['age_at_admit']

print(
    f'\nAge at admission (hospital): {hosp_age.mean():.1f} ± {hosp_age.std():.1f}  [median {hosp_age.median():.0f}]'
)
print(
    f'Age at admission (ICU):      {icu_age.mean():.1f} ± {icu_age.std():.1f}  [median {icu_age.median():.0f}]'
)

#  Gender
# Gender per admission
hosp_gender = admissions.merge(
    patients[['subject_id', 'gender']], on='subject_id', how='left', suffixes=('', '_pat')
)
n_female_hosp = (hosp_gender['gender'] == 'F').sum()
pct_female_hosp = n_female_hosp / len(hosp_gender) * 100

icu_gender = icu_admit.merge(
    patients[['subject_id', 'gender']], on='subject_id', how='left', suffixes=('', '_pat')
)
n_female_icu = (icu_gender['gender'] == 'F').sum()
pct_female_icu = n_female_icu / len(icu_gender) * 100

print(f'\nFemale admissions (hospital): {n_female_hosp:,} ({pct_female_hosp:.1f}%)')
print(f'Female admissions (ICU):      {n_female_icu:,} ({pct_female_icu:.1f}%)')

#  Length of stay
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_days'] = (
    admissions['dischtime'] - admissions['admittime']
).dt.total_seconds() / 86400

hosp_los = admissions['los_days']
icu_los = icustays['los']  # already in days

print(
    f'\nLOS hospital (days):  {hosp_los.mean():.1f} ± {hosp_los.std():.1f}  [median {hosp_los.median():.1f}]'
)
print(
    f'LOS ICU (days):       {icu_los.mean():.1f} ± {icu_los.std():.1f}  [median {icu_los.median():.1f}]'
)

#  Mortality
n_hosp_death = admissions['hospital_expire_flag'].sum()
pct_hosp_death = n_hosp_death / len(admissions) * 100

# ICU mortality: hospital_expire_flag for admissions that include an ICU stay
n_icu_death = icu_admit['hospital_expire_flag'].sum()
pct_icu_death = n_icu_death / len(icu_admit) * 100

print(f'\nIn-hospital mortality:  {n_hosp_death:,} ({pct_hosp_death:.1f}%)')
print(f'ICU mortality:          {n_icu_death:,} ({pct_icu_death:.1f}%)')

#  Notes
n_discharge = len(discharge)
n_radiology = len(radiology)

print(f'\nDischarge summaries:  {n_discharge:,}')
print(f'Radiology reports:    {n_radiology:,}')

#  Summary for slide
print(f'\n{"=" * 50}')
print('Slide table values:')
print(f'{"=" * 50}')
print(f'["Stays", "{n_admissions:,}", "{n_icu_stays:,}"],')
print(f'["Unique patients", "{n_hosp_patients:,}", "{n_icu_patients:,}"],')
print(
    f'["Age, mean (SD)", "{hosp_age.mean():.1f} ({hosp_age.std():.1f})", "{icu_age.mean():.1f} ({icu_age.std():.1f})"],'
)
print(
    f'["Female, n (%)", "{n_female_hosp:,} ({pct_female_hosp:.1f})", "{n_female_icu:,} ({pct_female_icu:.1f})"],'
)
print(
    f'["Length of stay, mean (SD)", "{hosp_los.mean():.1f} ({hosp_los.std():.1f})", "{icu_los.mean():.1f} ({icu_los.std():.1f})"],'
)
print(
    f'["In-hospital mortality, n (%)", "{n_hosp_death:,} ({pct_hosp_death:.1f})", "{n_icu_death:,} ({pct_icu_death:.1f})"],'
)
print(f'["Discharge summaries", "{n_discharge:,}", "\\u2014"],')
print(f'["Radiology reports", "{n_radiology:,}", "\\u2014"],')
