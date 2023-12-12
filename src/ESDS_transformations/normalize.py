"""Normalize the numerical values in a dataset to have zero mean and unit variance."""

from typing import Any

import numpy as np

from .base_class import BATCH_T, PT_DATA_T, ESDSTransformationFntr


class NormalizeFntr(ESDSTransformationFntr):
    """Join a dataset to a cohort with task labels."""

    def __init__(self, norm_params: dict[str, tuple[float, float]], **kwargs):
        """Initialize the join cohort function transformer.

        Args:
            cohort_df: A dataframe with columns patient_id, end_time (inclusive) and any optional label
                columns.
        """
        super().__init__(**kwargs)
        self.norm_params = norm_params

    def normalize_measurement(self, meas: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single measurement."""
        code = meas["code"]
        value = meas["numeric_value"]
        if code not in self.norm_params or value is None or value is np.nan:
            return meas
        mean, std = self.norm_params[code]
        meas["numeric_value"] = (value - mean) / std
        return meas

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        """Join the patient to the cohort dataframe."""
        patient["static_measurements"] = [
            self.normalize_measurement(m) for m in patient["static_measurements"]
        ]
        for e in patient["events"]:
            e["measurements"] = [self.normalize_measurement(m) for m in e["measurements"]]
        return {k: [v] for k, v in patient.items()}
