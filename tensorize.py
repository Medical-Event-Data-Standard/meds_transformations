import torch

from .base_class import PT_DATA_T, BATCH_DATA_T, ESDSTransformationFntr

class TensorizeFntr(ESDSTransformationFntr):
    def __init__(
        self,
        code_idxmap: dict[str, int],
        pad_sequences_to: int | None = None,
        pad_measurements_to: int | None = None,
    ):
        self.code_idxmap = code_idxmap
        self.pad_sequences_to = pad_sequences_to
        self.pad_measurements_to = pad_measurements_to

    @staticmethod
    def to_float(v: float | None) -> float:
        if v is None: return 0
        else: return v

    def tensorize_measurements_no_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        codes = torch.LongTensor([self.code_idxmap.get(m["code"], 0) for m in measurements])
        vals  = torch.FloatTensor([self.to_float(m["numeric_value"]) for m in measurements])
        vals_mask = torch.BoolTensor([m["numeric_value"] not in (None, float('nan')) for m in measurements])
        return codes, vals, vals_mask

    def tensorize_measurements_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        codes, vals, vals_mask = self.tensorize_measurements_no_pad(measurements)

        L = pad_measurements_to - len(codes)

        codes = torch.nn.functional.pad(codes, (0, L), value=0)
        vals = torch.nn.functional.pad(vals, (0, L), value=0)
        vals_mask = torch.nn.functional.pad(vals_mask, (0, L), value=False)

        return codes, vals, vals_mask

    if pad_measurements_to is None:
        tensorize_measurements = tensorize_measurements_no_pad
    else:
        tensorize_measurements = tensorize_measurements_pad

    def tensorize_subj_no_pad(self, subj: PT_DATA_T) -> PT_DATA_T:
        out_subj = {}

        static_keys = set(subj.keys()) - self.DYNAMIC_KEYS

        static_ids, static_values, static_values_mask = tensorize_measurements(
            subj["static_measurements"]
        )
        out_subj["static_ids"] = static_ids
        out_subj["static_values"] = static_values
        out_subj["static_values_mask"] = static_values_mask

        for k in set(subj.keys()) - {"events", "subject_id", "static_measurements"}:
            out_subj[k] = torch.tensor(subj[k])

        time = []
        dynamic_ids = []
        dynamic_values = []
        dynamic_values_mask = []
        time_derived_ids = []
        time_derived_values = []
        time_derived_values_mask = []

        st_time = subj["events"][0]["datetime"]

        other_per_event = defaultdict(list)

        for event in subj["events"]:
            time.append((event["datetime"] - st_time).total_seconds() / 60.0)
            i, v, m = tensorize_measurements(event["measurements"])
            dynamic_ids.append(i)
            dynamic_values.append(v)
            dynamic_values_mask.append(m)

            covered_keys = {"datetime", "measurements"}

            if "time_derived_measurements" in event:
                i, v, m = tensorize_measurements(event["time_derived_measurements"])
                time_derived_ids.append(i)
                time_derived_values.append(v)
                time_derived_values_mask.append(m)
                covered_keys.add("time_derived_measurements")


            for k in set(event.keys()) - covered_keys:
                other_per_event[k].append(event[k])

        out_subj["time"] = torch.FloatTensor(time)
        out_subj["dynamic_ids"] = dynamic_ids
        out_subj["dynamic_values"] = dynamic_values
        out_subj["dynamic_values_mask"] = dynamic_values_mask

        out_subj["time_derived_ids"] = time_derived_ids
        out_subj["time_derived_values"] = time_derived_values
        out_subj["time_derived_values_mask"] = time_derived_values_mask

        for k, v in other_per_event.items():
            out_subj[f"dynamic_{k}"] = torch.tensor(v)

        return out_subj

    def tensorize_subj_pad(subj: PT_DATA_T) -> PT_DATA_T:
        tensorized_subj = tensorize_subj_no_pad(subj)

        static_keys = [
            k for k in tensorized_subj.keys() if not k.startswith("dynamic_") or k.startswith("time")
        ]
        dynamic_keys = [k for k in tensorized_subj.keys() if k.startswith("dynamic_")]
        time_derived_keys = [k for k in tensorized_subj.keys() if k.startswith("time_derived_")]

        L = pad_sequences_to - len(tensorized_subj["time"])
        tensorized_subj["event_mask"] = torch.BoolTensor(
            ([True] * len(tensorized_subj["time"])) + ([False] * L)
        )
        tensorized_subj["time"] = torch.nn.functional.pad(
            tensorized_subj["time"], (0, L), value=0
        )

        for k in dynamic_keys + time_derived_keys:
            tensorized_subj[k] = torch.nn.functional.pad(
                torch.cat([T.unsqueeze(0) for T in tensorized_subj[k]]),
                (0, L),
                value=0,
            )
        return tensorized_subj

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        if if self.pad_sequences_to is None:
            tensorized = self.tensorize_subj_no_pad(patient)
        else:
            tensorized = self.tensorize_subj_pad(patient)

        return {k: [v] for k, v in tensorized.items()}
