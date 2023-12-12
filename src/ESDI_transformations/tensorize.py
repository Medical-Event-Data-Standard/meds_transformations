from collections import defaultdict
from typing import Any

import torch

from .base_class import BATCH_T, PT_DATA_T, ESDITransformationFntr


class TensorizeFntr(ESDITransformationFntr):
    def __init__(
        self,
        code_idxmap: dict[str, int],
        pad_sequences_to: int | None = None,
        pad_measurements_to: int | None = None,
    ):
        self.code_idxmap = code_idxmap
        self.pad_sequences_to = pad_sequences_to
        self.pad_measurements_to = pad_measurements_to

        if pad_measurements_to is None:
            self.tensorize_measurements = self.tensorize_measurements_no_pad
        else:
            self.tensorize_measurements = self.tensorize_measurements_pad

        if self.pad_sequences_to is None:
            self.tensorize_subject = self.tensorize_subj_no_pad
        else:
            self.tensorize_subject = self.tensorize_subj_pad

    @staticmethod
    def to_float(v: float | None) -> float:
        if v is None:
            return 0
        else:
            return v

    def tensorize_measurements_no_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        codes = torch.LongTensor([self.code_idxmap.get(m["code"], 0) for m in measurements])
        vals = torch.FloatTensor([self.to_float(m["numeric_value"]) for m in measurements])
        vals_mask = torch.BoolTensor([m["numeric_value"] not in (None, float("nan")) for m in measurements])
        return codes, vals, vals_mask

    def tensorize_measurements_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        codes, vals, vals_mask = self.tensorize_measurements_no_pad(measurements)

        L = self.pad_measurements_to - len(codes)

        codes = torch.nn.functional.pad(codes, (0, L), value=0)
        vals = torch.nn.functional.pad(vals, (0, L), value=0)
        vals_mask = torch.nn.functional.pad(vals_mask, (0, L), value=False)

        return codes, vals, vals_mask

    def tensorize_subj_no_pad(self, subj: PT_DATA_T) -> PT_DATA_T:
        out_subj = {}

        static_ids, static_values, static_values_mask = self.tensorize_measurements(
            subj["static_measurements"]
        )
        out_subj["static_ids"] = static_ids
        out_subj["static_values"] = static_values
        out_subj["static_values_mask"] = static_values_mask

        for k in set(subj.keys()) - {"events", "patient_id", "static_measurements"}:
            if subj[k] is None:
                raise ValueError(f"Can't tensorize `None` value {subj[k]} at `subj[{k}]`")
            out_subj[k] = torch.tensor(subj[k])

        time = []
        dynamic_ids = []
        dynamic_values = []
        dynamic_values_mask = []

        st_time = subj["events"][0]["time"]

        other_per_event = defaultdict(list)

        for event in subj["events"]:
            time.append((event["time"] - st_time).total_seconds() / 60.0)
            i, v, m = self.tensorize_measurements(event["measurements"])
            dynamic_ids.append(i)
            dynamic_values.append(v)
            dynamic_values_mask.append(m)

            covered_keys = {"time", "measurements"}

            for k in set(event.keys()) - covered_keys:
                if event[k] is None:
                    raise ValueError(f"Can't tensorize `None` value {event[k]} at `event[{k}]`")
                other_per_event[k].append(event[k])

        out_subj["time"] = torch.FloatTensor(time)
        out_subj["dynamic_ids"] = dynamic_ids
        out_subj["dynamic_values"] = dynamic_values
        out_subj["dynamic_values_mask"] = dynamic_values_mask

        for k, v in other_per_event.items():
            out_subj[f"dynamic_{k}"] = torch.tensor(v)

        return out_subj

    def tensorize_subj_pad(self, subj: PT_DATA_T) -> PT_DATA_T:
        tensorized_subj = self.tensorize_subj_no_pad(subj)

        dynamic_keys = [k for k in tensorized_subj.keys() if k.startswith("dynamic_")]

        L = self.pad_sequences_to - len(tensorized_subj["time"])
        tensorized_subj["event_mask"] = torch.BoolTensor(
            ([True] * len(tensorized_subj["time"])) + ([False] * L)
        )
        tensorized_subj["time"] = torch.nn.functional.pad(tensorized_subj["time"], (0, L), value=0)

        for k in dynamic_keys:
            try:
                max_len = max(len(T) for T in tensorized_subj[k])
                padded_dynamic_Ts = [
                    torch.nn.functional.pad(T, (0, max_len - len(T)), value=0) for T in tensorized_subj[k]
                ]
                tensorized_subj[k] = torch.nn.functional.pad(
                    torch.cat([T.unsqueeze(0) for T in padded_dynamic_Ts]),
                    (0, L),
                    value=0,
                )
            except Exception as e:
                raise RuntimeError(f"Error while padding {k}") from e
        return tensorized_subj

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        return {k: [v] for k, v in self.tensorize_subject(patient).items()}
