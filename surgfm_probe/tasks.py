"""Frame-level classification tasks of the benchmark suite, with the settings used for the reported scores.

Annotations are not distributed with this package. Build a manifest for each split (see the README)
and pass it together with the preset: ``surgfm-probe probe --task hyperkvasir ...``.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    column: str  # column abbreviation in the report tables
    num_classes: int
    selection_split: str  # split on which the best head is picked: "val", or "test" if the benchmark has no val split
    multi_label: bool = False

    @property
    def metric(self) -> str:
        return "f1_macro" if self.multi_label else "accuracy"


def _t(key, name, column, num_classes, selection_split, multi_label=False) -> Task:
    return Task(key, name, column, num_classes, selection_split, multi_label)


TASKS: Dict[str, Task] = {
    t.key: t
    for t in (
        _t("cholec80-phase", "Cholec80 phase", "ChPh", 7, "test"),
        _t("autolaparo-phase", "AutoLaparo phase", "AuPh", 7, "val"),
        _t("multibypass-phase", "MultiBypass140 phase", "MbPh", 12, "val"),
        _t("multibypass-step", "MultiBypass140 step", "MbSt", 46, "val"),
        _t("pitvis-step", "PitVis step", "PitV", 15, "val"),
        _t("hyperkvasir", "HyperKvasir", "HypK", 23, "val"),
        _t("cholecinstanceseg", "CholecInstanceSeg instrument", "ChIS", 7, "test"),
        _t("heichole", "HeiChole instrument", "HeiC", 7, "test"),
        _t("sarrarp50", "SAR-RARP50 gesture", "SARr", 7, "test"),
        _t("surgvu", "SurgVU instrument", "SgVU", 20, "test"),
        _t("lapgckvad", "LapGC-KVAD-30 anatomy", "LapG", 15, "test"),
    )
}


def get_task(key: str) -> Task:
    try:
        return TASKS[key]
    except KeyError:
        raise ValueError(f"unknown task {key!r}; choose from {sorted(TASKS)}") from None


def format_tasks() -> str:
    rows = [f"{'task':<20} {'classes':>7}  {'selected on':<11} {'metric':<9} name"]
    for t in TASKS.values():
        rows.append(f"{t.key:<20} {t.num_classes:>7}  {t.selection_split:<11} {t.metric:<9} {t.name}")
    rows.append("")
    rows.append(
        "'selected on test' tasks have no separate validation split: the head is chosen on the same\n"
        "images that are reported, so those scores are optimistic (pass --select-on-test)."
    )
    return "\n".join(rows)
