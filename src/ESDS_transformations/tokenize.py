"""Normalize the numerical values in a dataset to have zero mean and unit variance."""

from typing import Any

from .base_class import BATCH_T, PT_DATA_T, ESDSTransformationFntr


class TokenizeFntr(ESDSTransformationFntr):
    """Join a dataset to a cohort with task labels."""

    def __init__(self, vocab: list[str], **kwargs):
        """Initialize the join cohort function transformer.

        Args:
            cohort_df: A dataframe with columns patient_id, end_time (inclusive) and any optional label
                columns.
        """
        super().__init__(**kwargs)
        self.vocab = set(vocab)

    def tokenize_measurements(self, meas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize a single measurement."""
        return [m for m in meas if m["code"] in self.vocab]

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        """Join the patient to the cohort dataframe."""
        patient["static_measurements"] = self.tokenize_measurements(patient["static_measurements"])
        for e in patient["events"]:
            e["measurements"] = self.tokenize_measurements(e["measurements"])
        return [patient]
