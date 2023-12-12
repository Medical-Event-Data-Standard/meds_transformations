"""A base class underwriting other ESDS transformation functions."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

PT_DATA_T = dict[str, Any]
BATCH_T = dict[str, list[PT_DATA_T]]


class ESDSTransformationFntr(ABC):
    """Base class for ESDS transformation functions."""

    DYNAMIC_KEYS = {"events"}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def __transform_patient__(self, patient: PT_DATA_T) -> BATCH_T:
        raise NotImplementedError("Must implement __transform_patient__ method!")

    def __call__(self, batch: BATCH_T | PT_DATA_T) -> BATCH_T:
        if type(batch["patient_id"]) is not list:
            return self.__transform_patient__(batch)

        out_batch = defaultdict(list)
        batch_keys = list(batch.keys())
        as_pt_list = ({k: v for k, v in zip(batch_keys, vs)} for vs in zip(*[batch[k] for k in batch_keys]))

        for pt in as_pt_list:
            pt = self.__transform_patient__(pt)
            for k, v in pt.items():
                out_batch[k].extend(v)

        return dict(out_batch)
