import numpy as np

from .base_class import BATCH_T, PT_DATA_T, ESDSTransformationFntr


class SampleSubsequencesFntr(ESDSTransformationFntr):
    def __init__(
        self,
        max_seq_len: int = 256,
        n_samples_per_patient: int = 1,  # Should be set to approximately avg(pt_seq_len/max_seq_len)
        sample_strategy: str = "to_end",
    ):
        self.max_seq_len = max_seq_len
        match sample_strategy.lower():
            case "even":

                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    step_size = int(round(max_valid_st_idx / n_samples_per_patient))
                    return list(np.arange(0, max_valid_st_idx, step_size))

            case "random":

                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return list(np.random.choice(max_valid_st_idx, size=n_samples_per_patient))

            case "from_start":
                if n_samples_per_patient != 1:
                    raise ValueError(
                        f"Sampling with {sample_strategy} is only valid when n_samples_per_patient is 1"
                    )

                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return [0]

            case "to_end":
                if n_samples_per_patient != 1:
                    raise ValueError(
                        f"Sampling with {sample_strategy} is only valid when n_samples_per_patient is 1"
                    )

                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return [max_valid_st_idx]

            case _:
                raise ValueError(f"sample_strategy {sample_strategy} invalid.")

        self.sample_fn = sample_fn

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        static_keys = set(patient.keys()) - self.DYNAMIC_KEYS

        seq_len = len(patient["events"])
        if seq_len <= self.max_seq_len:
            return {k: [v] for k, v in patient.items()}

        out_batch = {k: [] for k in patient.keys()}
        st_indices = self.sample_fn(seq_len - self.max_seq_len)
        for k in static_keys:
            out_batch[k].extend([patient[k]] * len(st_indices))

        for st_idx in st_indices:
            for k in self.DYNAMIC_KEYS:
                out_batch[k].append(patient[k][st_idx : st_idx + self.max_seq_len])

        return out_batch
