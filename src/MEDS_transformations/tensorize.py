from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .base_class import BATCH_T, PT_DATA_T, MEDSTransformationFntr


class TensorizeFntr(MEDSTransformationFntr):
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
        """Converts a None to 0 otherwise returns the float value.

        Args:
            v: A float or `None`, which should be converted.

        Returns: 0 if ``v`` is `None` otherwise ``v``.

        Examples:
            >>> TensorizeFntr.to_float(1.0)
            1.0
            >>> TensorizeFntr.to_float(None)
            0
        """
        if v is None:
            return 0
        else:
            return v

    @staticmethod
    def is_present_float(v: float | None) -> bool:
        """Returns whether a float is not None.

        Args:
            v: A float or `None`, which should be checked.

        Returns: ``True`` if ``v`` is not `None` otherwise ``False``.

        Examples:
            >>> TensorizeFntr.is_present_float(1.0)
            True
            >>> TensorizeFntr.is_present_float(0)
            True
            >>> TensorizeFntr.is_present_float(np.float32(0.0))
            True
            >>> TensorizeFntr.is_present_float(np.uint32(3))
            True
            >>> TensorizeFntr.is_present_float(None)
            False
            >>> TensorizeFntr.is_present_float(np.NaN)
            False
            >>> TensorizeFntr.is_present_float(float("nan"))
            False
            >>> TensorizeFntr.is_present_float(float("inf"))
            False
            >>> TensorizeFntr.is_present_float(-float("inf"))
            False
        """
        match v:
            case None:
                return False
            case float() | int() | np.float32() | np.float64() | np.int32() | np.int64():
                return not (np.isnan(v) or np.isinf(v))
            case _:
                try:
                    return TensorizeFntr.is_present_float(float(v))
                except Exception:
                    return False

    def tensorize_measurements_no_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """Tensorizes a list of measurements into codes, values, and values-mask tensors.

        Args:
            measurements: A sequence of measurements in dictionary form.

        Returns: A tuple of three tensors: codes, containing the integer indices corresponding to the
            categorical codes in the measurements ``'code'`` key (or 0 if the code is not found), numerical
            values corresponding to the values in the measurements ``'numeric_value'`` field, and a boolean
            mask indicating whether values were present or not.

        Examples:
            >>> fntr = TensorizeFntr({"A": 1, "B": 2})
            >>> codes, values, values_mask = fntr.tensorize_measurements_no_pad([
            ...     {"code": "A", "numeric_value": 0.0},
            ...     {"code": "A"},
            ...     {"code": "B", "numeric_value": 2.0},
            ...     {"code": "B", "numeric_value": None},
            ...     {"code": "C", "numeric_value": float('nan')},
            ...     {"code": "A", "numeric_value": float('inf')},
            ... ])
            >>> codes
            tensor([1, 1, 2, 2, 0, 1])
            >>> values
            tensor([0., 0., 2., 0., nan, inf])
            >>> values_mask
            tensor([ True, False,  True, False, False, False])
        """

        codes = torch.LongTensor([self.code_idxmap.get(m["code"], 0) for m in measurements])
        vals = torch.FloatTensor([self.to_float(m.get("numeric_value", None)) for m in measurements])
        vals_mask = torch.BoolTensor(
            [self.is_present_float(m.get("numeric_value", None)) for m in measurements]
        )
        return codes, vals, vals_mask

    def tensorize_measurements_pad(
        self, measurements: list[dict[str, Any]]
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """Tensorizes a list of measurements into codes, values, and values-mask tensors, padded to max
        length.

        Args:
            measurements: A sequence of measurements in dictionary form.

        Returns: A tuple of three tensors: codes, containing the integer indices corresponding to the
            categorical codes in the measurements ``'code'`` key (or 0 if the code is not found), numerical
            values corresponding to the values in the measurements ``'numeric_value'`` field, and a boolean
            mask indicating whether values were present or not. All tensors are padded to the maximum length
            of the class object.

        Examples:
            >>> fntr = TensorizeFntr({"A": 1, "B": 2}, pad_measurements_to=5)
            >>> codes, values, values_mask = fntr.tensorize_measurements_pad([
            ...     {"code": "A", "numeric_value": 0},
            ...     {"code": "B", "numeric_value": 3.0},
            ...     {"code": "B", "numeric_value": None},
            ... ])
            >>> codes
            tensor([1, 2, 2, 0, 0])
            >>> values
            tensor([0., 3., 0., 0., 0.])
            >>> values_mask
            tensor([ True,  True, False, False, False])
        """
        codes, vals, vals_mask = self.tensorize_measurements_no_pad(measurements)

        L = self.pad_measurements_to - len(codes)

        codes = torch.nn.functional.pad(codes, (0, L), value=0)
        vals = torch.nn.functional.pad(vals, (0, L), value=0)
        vals_mask = torch.nn.functional.pad(vals_mask, (0, L), value=False)

        return codes, vals, vals_mask

    def tensorize_subj_no_pad(self, subj: PT_DATA_T) -> PT_DATA_T:
        """Tensorizes a subject into padded tensors of time_deltas & nested codes, values, and values-masks.

        Args:
            subj: A subject in dictionary form.

        Returns: A dictionary with the following keys and values:
          * ``static_ids``: A tensor of the static code indices.
          * ``static_values``: A tensor of the numeric values associated with static measurements.
          * ``static_values_mask``: A tensor of booleans indicating whether numeric static values were
            present.
          * ``time_delta``: A tensor of the time deltas between measurements. A delta encoding is used to
            avoid numerical overflow issues for long time series.
          * ``dynamic_ids``: A nested, _non-padded_ tensor of the dynamic code indices.
          * ``dynamic_values``: A nested, _non-padded_ tensor of the numeric values associated with dynamic
            measurements.
          * ``dynamic_values_mask``: A nested, _non-padded_ tensor of whether or not numeric values were
            present for dynamic measurements.

        Examples:
            >>> from datetime import datetime
            >>> fntr = TensorizeFntr({"A": 1, "B": 2, "D": 3})
            >>> out_subj = fntr.tensorize_subj_no_pad({
            ...     "patient_id": 3,
            ...     "static_measurements": [
            ...         {"code": "A"},
            ...         {"code": "B"},
            ...         {"code": "C", "numeric_value": 3.2},
            ...     ],
            ...     "events": [
            ...         {
            ...             "time": datetime(2023, 1, 1),
            ...             "measurements": [
            ...                 {"code": "A", "numeric_value": 0},
            ...                 {"code": "D", "numeric_value": 3.0},
            ...                 {"code": "B", "numeric_value": None},
            ...                 {"code": "A", "numeric_value": 10},
            ...             ],
            ...         }, {
            ...             "time": datetime(2023, 1, 2),
            ...             "measurements": [],
            ...         }, {
            ...             "time": datetime(2023, 1, 5),
            ...             "measurements": [
            ...                 {"code": "A", "numeric_value": 0},
            ...                 {"code": "D", "numeric_value": 1.0},
            ...             ],
            ...         },
            ...     ],
            ... })
            >>> assert (
            ...     set(out_subj.keys()) ==
            ...     {
            ...         "static_ids", "static_values", "static_values_mask", "time_delta", "dynamic_ids",
            ...         "dynamic_values", "dynamic_values_mask"
            ...     }
            ... ), f"Keys are wrong! Got {out_subj.keys()}"
            >>> out_subj["static_ids"]
            tensor([1, 2, 0])
            >>> out_subj["static_values"]
            tensor([0.0000, 0.0000, 3.2000])
            >>> out_subj["static_values_mask"]
            tensor([False, False,  True])
            >>> out_subj["time_delta"]
            tensor([   0., 1440., 4320.])
            >>> out_subj["dynamic_ids"]
            [tensor([1, 3, 2, 1]), tensor([], dtype=torch.int64), tensor([1, 3])]
            >>> out_subj["dynamic_values"]
            [tensor([ 0.,  3.,  0., 10.]), tensor([]), tensor([0., 1.])]
            >>> out_subj["dynamic_values_mask"]
            [tensor([ True,  True, False,  True]), tensor([], dtype=torch.bool), tensor([True, True])]
        """
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

        time_delta = []
        dynamic_ids = []
        dynamic_values = []
        dynamic_values_mask = []

        st_time = subj["events"][0]["time"]
        last_event_time = st_time

        other_per_event = defaultdict(list)

        for event in subj["events"]:
            time_delta.append((event["time"] - last_event_time).total_seconds() / 60.0)
            last_event_time = event["time"]
            i, v, m = self.tensorize_measurements(event["measurements"])
            dynamic_ids.append(i)
            dynamic_values.append(v)
            dynamic_values_mask.append(m)

            for k in set(event.keys()) - {"time", "measurements"}:
                if event[k] is None:
                    raise ValueError(f"Can't tensorize `None` value {event[k]} at `event[{k}]`")
                other_per_event[k].append(event[k])

        out_subj["time_delta"] = torch.FloatTensor(time_delta)
        out_subj["dynamic_ids"] = dynamic_ids
        out_subj["dynamic_values"] = dynamic_values
        out_subj["dynamic_values_mask"] = dynamic_values_mask

        for k, v in other_per_event.items():
            out_subj[f"dynamic_{k}"] = torch.tensor(v)

        return out_subj

    def tensorize_subj_pad(self, subj: PT_DATA_T) -> PT_DATA_T:
        """Tensorizes a subject into padded tensors of time_deltas & nested codes, values, and values-masks.

        Args:
            subj: A subject in dictionary form.

        Returns: A dictionary with the following keys and values:
          * ``static_ids``: A tensor of the static code indices.
            This may or may not be padded to a normalized length, depending on `self.pad_measurements_to`.
          * ``static_values``: A tensor of the numeric values associated with static measurements.
            This may or may not be padded to a normalized length, depending on `self.pad_measurements_to`.
          * ``static_values_mask``: A tensor of booleans indicating whether numeric static values were
            present.
            This may or may not be padded to a normalized length, depending on `self.pad_measurements_to`.
          * ``time_delta``: A tensor of the time deltas between measurements. A delta encoding is used to
            avoid numerical overflow issues for long time series.
            This will be padded to `self.pad_sequences_to`.
          * ``event_mask``: A boolean tensor indicating whether or not an event was present at a given
            sequence index.
          * ``dynamic_ids``: A padded tensor of the dynamic code indices.
            The outer tensor will be padded to `self.pad_sequences_to`.
            The inner tensor may or may not be padded to a normalized length, depending on
            `self.pad_measurements_to`.
          * ``dynamic_values``: A padded tensor of the numeric values associated with dynamic
            measurements.
            The outer tensor will be padded to `self.pad_sequences_to`.
            The inner tensor may or may not be padded to a normalized length, depending on
            `self.pad_measurements_to`.
          * ``dynamic_values_mask``: A padded tensor of whether or not numeric values were
            present for dynamic measurements.
            The outer tensor will be padded to `self.pad_sequences_to`.
            The inner tensor may or may not be padded to a normalized length, depending on
            `self.pad_measurements_to`.

        Examples:
            >>> from datetime import datetime
            >>> fntr = TensorizeFntr({"A": 1, "B": 2, "D": 3}, pad_sequences_to=4, pad_measurements_to=4)
            >>> out_subj = fntr.tensorize_subj_pad({
            ...     "patient_id": 3,
            ...     "static_measurements": [
            ...         {"code": "A"},
            ...         {"code": "B"},
            ...         {"code": "C", "numeric_value": 3.2},
            ...     ],
            ...     "events": [
            ...         {
            ...             "time": datetime(2023, 1, 1),
            ...             "measurements": [
            ...                 {"code": "A", "numeric_value": 0},
            ...                 {"code": "D", "numeric_value": 3.0},
            ...                 {"code": "B", "numeric_value": None},
            ...                 {"code": "A", "numeric_value": 10},
            ...             ],
            ...         }, {
            ...             "time": datetime(2023, 1, 2),
            ...             "measurements": [],
            ...         }, {
            ...             "time": datetime(2023, 1, 5),
            ...             "measurements": [
            ...                 {"code": "A", "numeric_value": 0},
            ...                 {"code": "D", "numeric_value": 1.0},
            ...             ],
            ...         },
            ...     ],
            ... })
            >>> assert (
            ...     set(out_subj.keys()) ==
            ...     {
            ...         "static_ids", "static_values", "static_values_mask", "time_delta", "dynamic_ids",
            ...         "dynamic_values", "dynamic_values_mask", "event_mask",
            ...     }
            ... ), f"Keys are wrong! Got {out_subj.keys()}"
            >>> out_subj["static_ids"]
            tensor([1, 2, 0, 0])
            >>> out_subj["static_values"]
            tensor([0.0000, 0.0000, 3.2000, 0.0000])
            >>> out_subj["static_values_mask"]
            tensor([False, False,  True, False])
            >>> out_subj["time_delta"]
            tensor([   0., 1440., 4320.,    0.])
            >>> out_subj["event_mask"]
            tensor([ True,  True,  True, False])
            >>> out_subj["dynamic_ids"]
            tensor([[1, 3, 2, 1],
                    [0, 0, 0, 0],
                    [1, 3, 0, 0],
                    [0, 0, 0, 0]])
            >>> out_subj["dynamic_values"]
            tensor([[ 0.,  3.,  0., 10.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]])
            >>> out_subj["dynamic_values_mask"]
            tensor([[ True,  True, False,  True],
                    [False, False, False, False],
                    [ True,  True, False, False],
                    [False, False, False, False]])
        """
        tensorized_subj = self.tensorize_subj_no_pad(subj)

        dynamic_keys = [k for k in tensorized_subj.keys() if k.startswith("dynamic_")]

        L = self.pad_sequences_to - len(tensorized_subj["time_delta"])
        tensorized_subj["event_mask"] = torch.BoolTensor(
            ([True] * len(tensorized_subj["time_delta"])) + ([False] * L)
        )
        tensorized_subj["time_delta"] = torch.nn.functional.pad(
            tensorized_subj["time_delta"], (0, L), value=0
        )

        for k in dynamic_keys:
            try:
                max_len = max(len(T) for T in tensorized_subj[k])
                padded_dynamic_Ts = [
                    torch.nn.functional.pad(T, (0, max_len - len(T)), value=0) for T in tensorized_subj[k]
                ]
                tensorized_subj[k] = torch.nn.functional.pad(
                    torch.cat([T.unsqueeze(0) for T in padded_dynamic_Ts]),
                    (0, 0, 0, L),
                    value=0,
                )
            except Exception as e:
                raise RuntimeError(f"Error while padding {k}") from e
        return tensorized_subj

    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        return {k: [v] for k, v in self.tensorize_subject(patient).items()}
