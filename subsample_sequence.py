class SampleStrategy(StrEnum):
    EVEN = auto()
    RANDOM = auto()
    FROM_START = auto()
    TO_END = auto()

from .base_class import PT_DATA_T, BATCH_DATA_T, ESDSTransformationFntr

class SubsampleSequenceFntr(ESDSTransformationFntr):
    def __init__(
        self,
        max_seq_len: int = 256,
        n_samples_per_patient: int = 1, # Should be set to approximately avg(pt_seq_len/max_seq_len)
        sample_strategy: SampleStrategy = SampleStrategy.RANDOM,
    ):
        match sample_strategy:
            case SampleStrategy.EVEN:
                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    step_size = int(round(max_valid_st_idx / n_samples_per_patient))
                    return list(np.arange(0, max_valid_st_idx, step_size))
            case SampleStrategy.RANDOM:
                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return list(np.random.choice(max_valid_st_idx, size=n_samples_per_patient))
            case SampleStrategy.FROM_START:
                if n_samples_per_patient != 1:
                    raise ValueError(
                        f"Sampling with {sample_strategy} is only valid when n_samples_per_patient is 1"
                    )
                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return [0]
            case SampleStrategy.TO_END:
                if n_samples_per_patient != 1:
                    raise ValueError(
                        f"Sampling with {sample_strategy} is only valid when n_samples_per_patient is 1"
                    )
                def sample_fn(max_valid_st_idx: int) -> list[int]:
                    return [max_valid_st_idx]
            case _: 
                raise ValueError(
                    f"sample_strategy {sample_strategy} invalid. Must be in {SampleStrategy.values()}"
                )

        self.sample_fn = sample_fn

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        seq_len = len(patient["events"])
        if seq_len <= max_seq_len:
            return {k: [v] for k, v in patient.items()}

        out_batch = {k: [] for k in patient.keys()}
        st_indices = self.sample_fn(seq_len - max_seq_len)
        for k in static_keys:
            out_batch[k].extend([patient[k]]*len(st_indices))

        for st_idx in st_indices:
            for k in dynamic_keys:
                out_batch[k].append(patient[k][st_idx:st_idx+max_seq_len])

        return out_batch
