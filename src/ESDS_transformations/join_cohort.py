"""Join a dataset to a cohort with task labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .base_class import BATCH_T, PT_DATA_T, ESDSTransformationFntr


class JoinCohortFntr(ESDSTransformationFntr):
    """Join a dataset to a cohort with task labels."""

    def __init__(self, cohort_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame | Path | str, **kwargs):
        """Initialize the join cohort function transformer.

        Args:
            cohort_df: A dataframe with columns patient_id, end_time (inclusive) and any optional label
                columns.
        """
        super().__init__(**kwargs)

        match cohort_df:
            case pd.DataFrame():
                self.cohort_df = pl.from_pandas(cohort_df).lazy()
            case pl.DataFrame():
                self.cohort_df = cohort_df.lazy()
            case pl.LazyFrame():
                pass
            case str() | Path() as fp:
                if type(fp) is str:
                    fp = Path(fp)

                if fp.suffix == ".csv":
                    self.cohort_df = pl.scan_csv(fp, infer_schema_length=1000)
                elif fp.suffix == ".parquet":
                    self.cohort_df = pl.scan_parquet(fp)
                else:
                    raise ValueError(f"Unrecognized file type {fp.suffix}!")
            case _:
                raise TypeError(f"Unrecognized type {type(cohort_df)} for cohort dataframe!")

        if not {"patient_id", "end_time"}.issubset(self.cohort_df.columns):
            raise KeyError("Missing mandatory cohort columns. Must have patient_id and end_time.")

        self.label_columns = sorted(list(set(self.cohort_df.columns) - {"patient_id", "end_time"}))

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        """Join the patient to the cohort dataframe."""

        cohort_rows = (
            self.cohort_df.filter(pl.col("patient_id") == patient["patient_id"]).collect().to_dicts()
        )

        out_batch = {k: [] for k in (list(patient.keys()) + self.label_columns)}

        static_keys = set(patient.keys()) - self.DYNAMIC_KEYS

        timestamps = np.array([e["time"] for e in patient["events"]])

        for row in cohort_rows:
            for k in static_keys:
                out_batch[k].append(patient[k])

            for k in self.label_columns:
                out_batch[k].append(row[k])

            end_idx = np.searchsorted(timestamps, row["end_time"], side="right")

            for k in self.DYNAMIC_KEYS:
                out_batch[k].append(patient[k][:end_idx])

        return out_batch
