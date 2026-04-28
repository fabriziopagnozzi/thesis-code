"""Add icd10_3char_list column to the LanceDB chunks table.

For each chunk, stores all distinct ICD-10 3-character codes diagnosed during
the corresponding hospital admission (hadm_id). This enables efficient
per-condition prefiltering in LanceDB.
"""

from typing import cast

import lancedb
import pyarrow as pa

from experiments.mimic.configs import global_cfg, setup_logging
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

COL_NAME = 'icd10_3char_list'


def build_hadm_to_icd(con) -> dict[int, list[str]]:
    """Return hadm_id → list[icd10_3char] for all admissions in unified_diagnoses."""
    rows = con.execute("""--sql
        SELECT hadm_id, list(DISTINCT LEFT(unified_icd10, 3)) AS icd10_3char_list
        FROM unified_diagnoses
        GROUP BY hadm_id
    """).fetchall()
    return {int(hadm_id): list_icd3_groups for (hadm_id, list_icd3_groups) in rows}


def add_icd_list_column(table_name: str = global_cfg.chunks_vec_table) -> None:
    db = lancedb.connect(MimicPaths.vector_db)
    table = db.open_table(table_name)

    if COL_NAME in table.schema.names:
        print(f"Column '{COL_NAME}' already exists in '{table_name}'. Nothing to do.")
        return

    full_data = table.to_arrow()
    con = connect_mimic_duckdb()
    hadm_to_icd = build_hadm_to_icd(con)

    hadm_ids = cast(list[int], full_data.column('hadm_id').to_pylist())
    icd_lists = [hadm_to_icd.get(h, []) for h in hadm_ids]
    new_col = pa.array(icd_lists, type=pa.list_(pa.string()))

    new_data = full_data.append_column(
        pa.field(COL_NAME, pa.list_(pa.string())),
        pa.chunked_array([new_col]),
    )

    db.create_table(table_name, new_data, mode='overwrite')
    print(f"Done. Column '{COL_NAME}' added to '{table_name}'.")


if __name__ == '__main__':
    setup_logging()
    add_icd_list_column()
