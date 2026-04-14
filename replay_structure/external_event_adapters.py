from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _coerce_epochs_dataframe(epochs: Any) -> pd.DataFrame:
    if isinstance(epochs, pd.DataFrame):
        epoch_df = epochs.copy()
    elif hasattr(epochs, "to_dataframe"):
        epoch_df = epochs.to_dataframe().copy()
    elif hasattr(epochs, "df"):
        epoch_df = epochs.df.copy()
    else:
        raise TypeError("epochs must be a pandas DataFrame or an Epoch-like object")
    required_columns = {"start", "stop"}
    if not required_columns.issubset(set(epoch_df.columns)):
        raise ValueError("epochs dataframe must contain 'start' and 'stop' columns")
    return epoch_df.reset_index(drop=True)


def _coerce_spikemats_to_dict(spikemats: Any) -> Dict[int, Any]:
    if isinstance(spikemats, dict):
        sorted_keys = sorted(spikemats.keys())
        return {i: spikemats[key] for i, key in enumerate(sorted_keys)}
    if isinstance(spikemats, (list, tuple)):
        return {i: spikemat for i, spikemat in enumerate(spikemats)}
    raise TypeError("spikemats must be a dict, list, or tuple")


def _coerce_time_window_ms(time_window_ms: Optional[int] = None, time_window_s: Optional[float] = None) -> int:
    if time_window_ms is None and time_window_s is None:
        raise ValueError("one of time_window_ms or time_window_s must be provided")
    if time_window_ms is None:
        time_window_ms = int(round(float(time_window_s) * 1000))
    if time_window_ms <= 0:
        raise ValueError("time_window_ms must be positive")
    return int(time_window_ms)


def _coerce_time_window_advance_ms(time_window_ms: int, time_window_advance_ms: Optional[int] = None, time_window_advance_s: Optional[float] = None) -> int:
    if time_window_advance_ms is None and time_window_advance_s is None:
        return int(time_window_ms)
    if time_window_advance_ms is None:
        time_window_advance_ms = int(round(float(time_window_advance_s) * 1000))
    if time_window_advance_ms <= 0:
        raise ValueError("time_window_advance_ms must be positive")
    return int(time_window_advance_ms)


def _infer_square_grid_shape(n_grid: int) -> Tuple[int, int]:
    n_bins_x = int(np.sqrt(n_grid))
    if n_bins_x * n_bins_x != n_grid:
        raise ValueError(
            "n_bins_x/n_bins_y must be provided when place fields are flattened and n_grid is not a perfect square"
        )
    return n_bins_x, n_bins_x


def _extract_place_fields_and_geometry(place_fields: Any) -> Tuple[np.ndarray, Optional[int], Optional[int], Optional[float], Optional[str]]:
    if isinstance(place_fields, np.ndarray):
        return place_fields, None, None, None, None

    if hasattr(place_fields, "ratemap") and hasattr(place_fields.ratemap, "tuning_curves"):
        tuning_curves = np.asarray(place_fields.ratemap.tuning_curves)
        inferred_layout = "cells_by_grid"
        if tuning_curves.ndim == 3:
            inferred_layout = "cells_by_x_by_y"
        n_bins_x = None
        n_bins_y = None
        bin_size_cm = None
        if hasattr(place_fields, "xbin") and getattr(place_fields, "xbin") is not None:
            xbin = np.asarray(place_fields.xbin)
            if xbin.ndim == 1 and len(xbin) >= 2:
                n_bins_x = len(xbin) - 1
                bin_size_cm = float(np.median(np.diff(xbin)))
        if hasattr(place_fields, "ybin") and getattr(place_fields, "ybin") is not None:
            ybin = np.asarray(place_fields.ybin)
            if ybin.ndim == 1 and len(ybin) >= 2:
                n_bins_y = len(ybin) - 1
                if bin_size_cm is None:
                    bin_size_cm = float(np.median(np.diff(ybin)))
        elif tuning_curves.ndim == 2:
            n_bins_y = 1
            if n_bins_x is None:
                n_bins_x = tuning_curves.shape[1]
        return tuning_curves, n_bins_x, n_bins_y, bin_size_cm, inferred_layout

    raise TypeError("place_fields must be a numpy array or a place-field object with ratemap.tuning_curves")


def _normalize_pf_matrix(place_fields: Any, n_bins_x: Optional[int] = None, n_bins_y: Optional[int] = None, bin_size_cm: Optional[float] = None, place_field_layout: Optional[str] = None) -> Tuple[np.ndarray, int, int, int]:
    pf_array, inferred_n_bins_x, inferred_n_bins_y, inferred_bin_size_cm, inferred_layout = _extract_place_fields_and_geometry(place_fields)
    if place_field_layout is None:
        place_field_layout = inferred_layout

    if place_field_layout == "cells_by_x_by_y":
        if pf_array.ndim != 3:
            raise ValueError("cells_by_x_by_y place fields must be a 3D array")
        pf_matrix = pf_array.reshape(pf_array.shape[0], -1)
        n_bins_x = n_bins_x or inferred_n_bins_x or pf_array.shape[1]
        n_bins_y = n_bins_y or inferred_n_bins_y or pf_array.shape[2]
    elif place_field_layout == "x_by_y_by_cells":
        if pf_array.ndim != 3:
            raise ValueError("x_by_y_by_cells place fields must be a 3D array")
        pf_matrix = np.moveaxis(pf_array, -1, 0).reshape(pf_array.shape[-1], -1)
        n_bins_x = n_bins_x or inferred_n_bins_x or pf_array.shape[0]
        n_bins_y = n_bins_y or inferred_n_bins_y or pf_array.shape[1]
    elif place_field_layout == "cells_by_grid":
        if pf_array.ndim != 2:
            raise ValueError("cells_by_grid place fields must be a 2D array")
        pf_matrix = pf_array
        if n_bins_x is None or n_bins_y is None:
            inferred_n_bins_x, inferred_n_bins_y = _infer_square_grid_shape(pf_array.shape[1])
            n_bins_x = n_bins_x or inferred_n_bins_x
            n_bins_y = n_bins_y or inferred_n_bins_y
    elif place_field_layout == "grid_by_cells":
        if pf_array.ndim != 2:
            raise ValueError("grid_by_cells place fields must be a 2D array")
        pf_matrix = pf_array.T
        if n_bins_x is None or n_bins_y is None:
            inferred_n_bins_x, inferred_n_bins_y = _infer_square_grid_shape(pf_matrix.shape[1])
            n_bins_x = n_bins_x or inferred_n_bins_x
            n_bins_y = n_bins_y or inferred_n_bins_y
    else:
        raise ValueError(f"unsupported place_field_layout: {place_field_layout}")

    if (n_bins_x is None) or (n_bins_y is None):
        raise ValueError("n_bins_x and n_bins_y must be resolved for external place fields")
    resolved_bin_size_cm = int(round(float(bin_size_cm if bin_size_cm is not None else inferred_bin_size_cm if inferred_bin_size_cm is not None else 4)))
    if pf_matrix.shape[1] != (int(n_bins_x) * int(n_bins_y)):
        raise ValueError("place field matrix width does not match n_bins_x * n_bins_y")
    return np.asarray(pf_matrix, dtype=float), int(n_bins_x), int(n_bins_y), resolved_bin_size_cm


def _normalize_spikemat(spikemat: Any, n_cells: int, spikemat_layout: str) -> Optional[np.ndarray]:
    if spikemat is None:
        return None
    spikemat = np.asarray(spikemat)
    if spikemat.ndim != 2:
        raise ValueError("each spikemat must be a 2D array")
    if spikemat_layout == "time_by_cells":
        normalized = spikemat
    elif spikemat_layout == "cells_by_time":
        normalized = spikemat.T
    else:
        raise ValueError(f"unsupported spikemat_layout: {spikemat_layout}")
    if normalized.shape[1] != n_cells:
        raise ValueError("spikemat width does not match number of cells in pf_matrix")
    return np.asarray(normalized, dtype=int)


def _summarize_canonical_payload(pf_matrix: np.ndarray, spikemats: Dict[int, Optional[np.ndarray]], time_window_ms: int, time_window_advance_ms: int, bin_size_cm: int, n_bins_x: int, n_bins_y: int, event_intervals_s: Optional[np.ndarray], source_name: Optional[str]) -> Dict[str, Any]:
    valid_spikemats = [spikemat for spikemat in spikemats.values() if spikemat is not None]
    if len(valid_spikemats) > 0:
        event_n_time_bins = np.array([spikemat.shape[0] for spikemat in valid_spikemats], dtype=int)
    else:
        event_n_time_bins = np.array([], dtype=int)
    event_duration_s = None
    if event_intervals_s is not None:
        event_duration_s = np.diff(event_intervals_s, axis=1).reshape(-1)
    return {
        "source_name": source_name,
        "n_events_total": len(spikemats),
        "n_events_valid": len(valid_spikemats),
        "n_cells": int(pf_matrix.shape[0]),
        "n_grid": int(pf_matrix.shape[1]),
        "n_bins_x": int(n_bins_x),
        "n_bins_y": int(n_bins_y),
        "bin_size_cm": int(bin_size_cm),
        "time_window_ms": int(time_window_ms),
        "time_window_advance_ms": int(time_window_advance_ms),
        "has_overlapping_time_bins": bool(time_window_advance_ms != time_window_ms),
        "event_n_time_bins_min": int(event_n_time_bins.min()) if len(event_n_time_bins) > 0 else 0,
        "event_n_time_bins_median": float(np.median(event_n_time_bins)) if len(event_n_time_bins) > 0 else 0.0,
        "event_n_time_bins_max": int(event_n_time_bins.max()) if len(event_n_time_bins) > 0 else 0,
        "event_duration_s_min": float(np.min(event_duration_s)) if event_duration_s is not None and len(event_duration_s) > 0 else None,
        "event_duration_s_median": float(np.median(event_duration_s)) if event_duration_s is not None and len(event_duration_s) > 0 else None,
        "event_duration_s_max": float(np.max(event_duration_s)) if event_intervals_s is not None and len(event_duration_s) > 0 else None,
    }


def build_canonical_payload(place_fields: Any, spikemats: Any, time_window_ms: Optional[int] = None, time_window_s: Optional[float] = None, bin_size_cm: Optional[float] = None, n_bins_x: Optional[int] = None, n_bins_y: Optional[int] = None, time_window_advance_ms: Optional[int] = None, time_window_advance_s: Optional[float] = None, event_intervals_s: Optional[np.ndarray] = None, place_field_layout: Optional[str] = None, spikemat_layout: str = "time_by_cells", source_name: Optional[str] = None, source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pf_matrix, n_bins_x, n_bins_y, bin_size_cm = _normalize_pf_matrix(
        place_fields,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        bin_size_cm=bin_size_cm,
        place_field_layout=place_field_layout,
    )
    time_window_ms = _coerce_time_window_ms(time_window_ms=time_window_ms, time_window_s=time_window_s)
    time_window_advance_ms = _coerce_time_window_advance_ms(
        time_window_ms,
        time_window_advance_ms=time_window_advance_ms,
        time_window_advance_s=time_window_advance_s,
    )
    spikemats_dict = _coerce_spikemats_to_dict(spikemats)
    normalized_spikemats = {
        event_id: _normalize_spikemat(spikemat, pf_matrix.shape[0], spikemat_layout)
        for event_id, spikemat in spikemats_dict.items()
    }
    if event_intervals_s is not None:
        event_intervals_s = np.asarray(event_intervals_s, dtype=float)
        if event_intervals_s.shape != (len(normalized_spikemats), 2):
            raise ValueError("event_intervals_s must have shape (n_events, 2)")
    diagnostics = _summarize_canonical_payload(
        pf_matrix,
        normalized_spikemats,
        time_window_ms,
        time_window_advance_ms,
        bin_size_cm,
        n_bins_x,
        n_bins_y,
        event_intervals_s,
        source_name,
    )
    payload = {
        "pf_matrix": pf_matrix,
        "spikemats": normalized_spikemats,
        "time_window_ms": time_window_ms,
        "time_window_advance_ms": time_window_advance_ms,
        "bin_size_cm": bin_size_cm,
        "n_bins_x": n_bins_x,
        "n_bins_y": n_bins_y,
        "event_intervals_s": event_intervals_s,
        "source_name": source_name,
        "source_metadata": dict(source_metadata or {}),
        "diagnostics": diagnostics,
    }
    return payload


def validate_canonical_payload(payload: Dict[str, Any], require_square_grid: bool = True) -> Dict[str, Any]:
    required_keys = {
        "pf_matrix",
        "spikemats",
        "time_window_ms",
        "time_window_advance_ms",
        "bin_size_cm",
        "n_bins_x",
        "n_bins_y",
    }
    missing_keys = required_keys.difference(payload.keys())
    if len(missing_keys) > 0:
        raise ValueError(f"canonical payload missing keys: {sorted(missing_keys)}")

    pf_matrix = np.asarray(payload["pf_matrix"])
    if pf_matrix.ndim != 2:
        raise ValueError("pf_matrix must be 2D")

    n_bins_x = int(payload["n_bins_x"])
    n_bins_y = int(payload["n_bins_y"])
    if require_square_grid and n_bins_x != n_bins_y:
        raise ValueError(
            "external structure input currently requires a square spatial grid because diffusion/momentum code assumes sqrt(n_grid) indexing"
        )
    if pf_matrix.shape[1] != (n_bins_x * n_bins_y):
        raise ValueError("pf_matrix width must equal n_bins_x * n_bins_y")

    spikemats = _coerce_spikemats_to_dict(payload["spikemats"])
    for event_id, spikemat in spikemats.items():
        if spikemat is None:
            continue
        spikemat = np.asarray(spikemat)
        if spikemat.ndim != 2:
            raise ValueError(f"spikemat {event_id} must be 2D")
        if spikemat.shape[1] != pf_matrix.shape[0]:
            raise ValueError(f"spikemat {event_id} width must match pf_matrix n_cells")
    return payload


def build_payload_from_neuropy_epochs_spkcount(epochs: Any, spkcount: Sequence[np.ndarray], place_fields: Any, decoding_time_bin_size_s: Optional[float] = None, decoding_time_bin_size_ms: Optional[int] = None, bin_size_cm: Optional[float] = None, n_bins_x: Optional[int] = None, n_bins_y: Optional[int] = None, slideby_s: Optional[float] = None, slideby_ms: Optional[int] = None, place_field_layout: Optional[str] = None, source_name: str = "neuropy_epochs_spkcount", source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    epoch_df = _coerce_epochs_dataframe(epochs)
    time_window_ms = _coerce_time_window_ms(
        time_window_ms=decoding_time_bin_size_ms,
        time_window_s=decoding_time_bin_size_s,
    )
    time_window_advance_ms = _coerce_time_window_advance_ms(
        time_window_ms,
        time_window_advance_ms=slideby_ms,
        time_window_advance_s=slideby_s,
    )
    return build_canonical_payload(
        place_fields=place_fields,
        spikemats=spkcount,
        time_window_ms=time_window_ms,
        time_window_advance_ms=time_window_advance_ms,
        bin_size_cm=bin_size_cm,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        event_intervals_s=epoch_df[["start", "stop"]].to_numpy(dtype=float),
        place_field_layout=place_field_layout or "cells_by_x_by_y",
        spikemat_layout="cells_by_time",
        source_name=source_name,
        source_metadata=source_metadata,
    )


def build_payload_from_pypho_decoded_epochs_result(decoded_epochs_result: Any, place_fields: Any, bin_size_cm: Optional[float] = None, n_bins_x: Optional[int] = None, n_bins_y: Optional[int] = None, place_field_layout: Optional[str] = None, source_name: str = "pypho_decoded_epochs_result", source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    epoch_df = _coerce_epochs_dataframe(decoded_epochs_result.filter_epochs)
    decoding_time_bin_size = getattr(decoded_epochs_result, "decoding_time_bin_size", None)
    decoding_slideby = getattr(decoded_epochs_result, "decoding_slideby", None)
    spkcount = getattr(decoded_epochs_result, "spkcount", None)
    if spkcount is None:
        raise ValueError("decoded_epochs_result must provide spkcount")
    return build_canonical_payload(
        place_fields=place_fields,
        spikemats=spkcount,
        time_window_s=decoding_time_bin_size,
        time_window_advance_s=decoding_slideby,
        bin_size_cm=bin_size_cm,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        event_intervals_s=epoch_df[["start", "stop"]].to_numpy(dtype=float),
        place_field_layout=place_field_layout,
        spikemat_layout="cells_by_time",
        source_name=source_name,
        source_metadata=source_metadata,
    )


def build_payload_from_replayswitchinghmm(place_fields: np.ndarray, replay_spikemats: Sequence[np.ndarray], time_window_ms: Optional[int] = None, time_window_s: Optional[float] = None, bin_size_cm: Optional[float] = None, n_bins_x: Optional[int] = None, n_bins_y: Optional[int] = None, time_window_advance_ms: Optional[int] = None, time_window_advance_s: Optional[float] = None, event_intervals_s: Optional[np.ndarray] = None, source_name: str = "replayswitchinghmm", source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return build_canonical_payload(
        place_fields=place_fields,
        spikemats=replay_spikemats,
        time_window_ms=time_window_ms,
        time_window_s=time_window_s,
        time_window_advance_ms=time_window_advance_ms,
        time_window_advance_s=time_window_advance_s,
        bin_size_cm=bin_size_cm,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        event_intervals_s=event_intervals_s,
        place_field_layout="grid_by_cells",
        spikemat_layout="time_by_cells",
        source_name=source_name,
        source_metadata=source_metadata,
    )
