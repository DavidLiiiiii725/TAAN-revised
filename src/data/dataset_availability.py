"""Dataset availability checking utilities.

Provides explicit availability checks for required datasets (AIME, GSM8K,
HumanEval, MMLU) with robust 401/authentication error handling and clear,
actionable status reporting.

Usage::

    from src.data.dataset_availability import check_required_datasets, print_availability_report

    results = check_required_datasets()
    print_availability_report(results)
    if any(r.status != DatasetStatus.AVAILABLE for r in results):
        sys.exit(1)
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required dataset registry
# ---------------------------------------------------------------------------

#: Default set of datasets to check.  Each entry maps a short *label* to a
#: dict with the HuggingFace ``name``, ``split``, and optional ``config``
#: (subset name).
REQUIRED_DATASETS: Dict[str, Dict] = {
    "aime": {
        "name": "Maxwell-Jia/AIME_1983_2024",
        "split": "train",
        "config": None,
    },
    "gsm8k": {
        "name": "openai/gsm8k",
        "split": "train",
        "config": "main",
    },
    "humaneval": {
        "name": "openai/humaneval",
        "split": "test",
        "config": None,
    },
    "mmlu": {
        "name": "cais/mmlu",
        "split": "test",
        "config": "all",
    },
}


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class DatasetStatus(str, enum.Enum):
    """Possible outcomes of a dataset availability check."""

    AVAILABLE = "available"
    AUTH_REQUIRED = "auth-required"
    NOT_FOUND = "not-found"
    OTHER_ERROR = "other-error"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetCheckResult:
    """Result of checking one dataset."""

    label: str                   # Short name, e.g. "gsm8k"
    dataset_id: str              # HF Hub identifier, e.g. "openai/gsm8k"
    split: str                   # Attempted split
    config: Optional[str]        # Attempted subset / config name
    status: DatasetStatus
    message: str = field(default="")


# ---------------------------------------------------------------------------
# Exception classifier
# ---------------------------------------------------------------------------

def classify_exception(exc: Exception) -> Tuple[DatasetStatus, str]:
    """Classify a ``datasets``/HuggingFace Hub exception into a :class:`DatasetStatus`.

    Checks are attempted in order:
    1. Specific HuggingFace Hub typed exceptions (``GatedRepoError``,
       ``RepositoryNotFoundError``, ``HfHubHTTPError``).
    2. String-based heuristics for HTTP status codes and keywords.

    Args:
        exc: The exception raised when trying to load a dataset.

    Returns:
        A ``(DatasetStatus, message)`` tuple.
    """
    exc_str = str(exc)

    # ------------------------------------------------------------------
    # 1. Try typed HF Hub exceptions (huggingface_hub >= 0.20 path)
    # ------------------------------------------------------------------
    try:
        from huggingface_hub.errors import (  # type: ignore[import]
            GatedRepoError,
            RepositoryNotFoundError,
        )
        if isinstance(exc, GatedRepoError):
            return DatasetStatus.AUTH_REQUIRED, exc_str
        if isinstance(exc, RepositoryNotFoundError):
            return DatasetStatus.NOT_FOUND, exc_str
    except (ImportError, AttributeError):
        pass

    # ------------------------------------------------------------------
    # 2. Try typed HF Hub exceptions (huggingface_hub < 0.20 path)
    # ------------------------------------------------------------------
    try:
        from huggingface_hub.utils import (  # type: ignore[import]
            GatedRepoError,
            RepositoryNotFoundError,
        )
        if isinstance(exc, GatedRepoError):
            return DatasetStatus.AUTH_REQUIRED, exc_str
        if isinstance(exc, RepositoryNotFoundError):
            return DatasetStatus.NOT_FOUND, exc_str
    except (ImportError, AttributeError):
        pass

    # ------------------------------------------------------------------
    # 3. HTTP-code and keyword heuristics
    # ------------------------------------------------------------------
    exc_lower = exc_str.lower()

    # Authentication / authorization failures
    if (
        "401" in exc_str
        or "unauthorized" in exc_lower
        or "403" in exc_str
        or "forbidden" in exc_lower
        or "gated" in exc_lower
        or ("access" in exc_lower and "restricted" in exc_lower)
        or "authentication" in exc_lower
        or ("token" in exc_lower and "required" in exc_lower)
    ):
        return DatasetStatus.AUTH_REQUIRED, exc_str

    # Not-found failures
    if (
        "404" in exc_str
        or "not found" in exc_lower
        or "doesn't exist" in exc_lower
        or "does not exist" in exc_lower
        or "no such" in exc_lower
    ):
        return DatasetStatus.NOT_FOUND, exc_str

    return DatasetStatus.OTHER_ERROR, exc_str


# ---------------------------------------------------------------------------
# Per-dataset check
# ---------------------------------------------------------------------------

def check_dataset_availability(
    dataset_id: str,
    split: str = "train",
    config: Optional[str] = None,
) -> Tuple[DatasetStatus, str]:
    """Check whether *dataset_id* is accessible on the HuggingFace Hub.

    Uses streaming mode so no data is actually downloaded; only the first
    record is fetched to confirm access.

    Args:
        dataset_id: HuggingFace dataset identifier, e.g. ``"openai/gsm8k"``.
        split:      Dataset split to attempt, e.g. ``"train"``.
        config:     Optional dataset subset/config name, e.g. ``"main"``.

    Returns:
        ``(DatasetStatus, message)`` tuple.  *message* is empty on success.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        return DatasetStatus.OTHER_ERROR, (
            "The `datasets` package is not installed. "
            "Install it with: pip install datasets"
        )

    try:
        load_kwargs: Dict = {"streaming": True}
        if config:
            ds = load_dataset(dataset_id, config, split=split, **load_kwargs)
        else:
            ds = load_dataset(dataset_id, split=split, **load_kwargs)

        # Iterate one record to confirm actual access rights.
        next(iter(ds))
        return DatasetStatus.AVAILABLE, ""

    except StopIteration:
        # Dataset is accessible but empty — still consider it available.
        return DatasetStatus.AVAILABLE, "(dataset is empty)"

    except Exception as exc:  # noqa: BLE001
        return classify_exception(exc)


# ---------------------------------------------------------------------------
# Batch check for required datasets
# ---------------------------------------------------------------------------

def check_required_datasets(
    datasets: Optional[Dict[str, Dict]] = None,
) -> List[DatasetCheckResult]:
    """Check availability of all required datasets.

    Args:
        datasets: Optional override for the dataset registry.  Defaults to
                  :data:`REQUIRED_DATASETS`.

    Returns:
        List of :class:`DatasetCheckResult` objects, one per dataset.
    """
    if datasets is None:
        datasets = REQUIRED_DATASETS

    results: List[DatasetCheckResult] = []
    for label, ds_cfg in datasets.items():
        ds_id: str = ds_cfg["name"]
        split: str = ds_cfg.get("split", "train")
        config: Optional[str] = ds_cfg.get("config")

        logger.info(
            "Checking availability: %s (%s, split=%s%s)",
            label.upper(),
            ds_id,
            split,
            f", config={config}" if config else "",
        )
        status, message = check_dataset_availability(ds_id, split=split, config=config)
        results.append(
            DatasetCheckResult(
                label=label,
                dataset_id=ds_id,
                split=split,
                config=config,
                status=status,
                message=message,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    DatasetStatus.AVAILABLE: "✓  AVAILABLE",
    DatasetStatus.AUTH_REQUIRED: "✗  AUTH-REQ ",
    DatasetStatus.NOT_FOUND: "✗  NOT-FOUND",
    DatasetStatus.OTHER_ERROR: "✗  ERROR    ",
}


def print_availability_report(results: List[DatasetCheckResult]) -> None:
    """Print a formatted availability report to *stdout*.

    For each dataset the report shows:
    - Status icon
    - Short label
    - Full HF dataset identifier
    - Attempted split/config
    - Actionable guidance for non-available datasets

    Args:
        results: List returned by :func:`check_required_datasets`.
    """
    sep = "=" * 65
    print()
    print(sep)
    print("  DATASET AVAILABILITY REPORT")
    print(sep)

    for r in results:
        config_str = f"  config={r.config}" if r.config else ""
        icon = _STATUS_ICON.get(r.status, "?")
        print(f"  [{icon}]  {r.label.upper():<12s}  {r.dataset_id}  split={r.split}{config_str}")

        if r.status == DatasetStatus.AUTH_REQUIRED:
            print("              → Authentication required.  Authenticate via one of:")
            print("                  huggingface-cli login")
            print("                  export HF_TOKEN=<your_token>")
            if "gated" in r.message.lower():
                print(f"                  Accept dataset terms at:")
                print(f"                  https://huggingface.co/datasets/{r.dataset_id}")
            if r.message:
                short = r.message[:200] + ("…" if len(r.message) > 200 else "")
                print(f"              → Detail: {short}")

        elif r.status == DatasetStatus.NOT_FOUND:
            print(f"              → Dataset '{r.dataset_id}' was not found on Hugging Face Hub.")
            print(f"                  Verify the dataset ID at: https://huggingface.co/datasets")

        elif r.status == DatasetStatus.OTHER_ERROR and r.message:
            short = r.message[:200] + ("…" if len(r.message) > 200 else "")
            print(f"              → Error: {short}")

    print(sep)

    statuses = [r.status for r in results]
    n_ok = statuses.count(DatasetStatus.AVAILABLE)
    n_auth = statuses.count(DatasetStatus.AUTH_REQUIRED)
    n_nf = statuses.count(DatasetStatus.NOT_FOUND)
    n_err = statuses.count(DatasetStatus.OTHER_ERROR)
    print(
        f"  Summary: {n_ok} available, {n_auth} auth-required, "
        f"{n_nf} not-found, {n_err} other-error"
    )
    print(sep)
    print()
