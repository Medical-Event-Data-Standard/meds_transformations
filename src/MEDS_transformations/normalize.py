"""Normalize the numerical values in a dataset to have zero mean and unit variance."""

from typing import Any

import numpy as np

from .base_class import BATCH_T, PT_DATA_T, MEDSTransformationFntr


class NormalizeFntr(MEDSTransformationFntr):
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
        """Normalize a single measurement to have zero mean and unit variance.

        Args:
            meas: A measurement dictionary with keys code and (optionally) numeric_value.

        Returns:
            The measurement dictionary with numeric_value normalized, if the code is in the norm_params and
            numeric_value is not None.

        Examples:
            >>> norm_params = {"HR": (70, 10)}
            >>> fntr = NormalizeFntr(norm_params)
            >>> fntr.normalize_measurement({"code": "HR", "numeric_value": 80})
            {'code': 'HR', 'numeric_value': 1.0}
            >>> fntr.normalize_measurement({"code": "HR", "numeric_value": 70})
            {'code': 'HR', 'numeric_value': 0.0}
            >>> fntr.normalize_measurement({"code": "HR", "numeric_value": 60})
            {'code': 'HR', 'numeric_value': -1.0}
            >>> fntr.normalize_measurement({"code": "HR", "numeric_value": None})
            {'code': 'HR', 'numeric_value': None}
            >>> fntr.normalize_measurement({"code": "HR"})
            {'code': 'HR'}
            >>> fntr.normalize_measurement({"code": "HR", "numeric_value": np.nan})
            {'code': 'HR', 'numeric_value': nan}
            >>> fntr.normalize_measurement({"code": "BP", "numeric_value": 80})
            {'code': 'BP', 'numeric_value': 80}
        """
        code = meas["code"]
        value = meas.get("numeric_value", None)
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
