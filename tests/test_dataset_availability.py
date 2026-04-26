"""
Unit tests for dataset availability checking (src/data/dataset_availability.py).

Covered scenarios
-----------------
1. classify_exception – 401 / Unauthorized → AUTH_REQUIRED
2. classify_exception – 403 / Forbidden   → AUTH_REQUIRED
3. classify_exception – gated keyword     → AUTH_REQUIRED
4. classify_exception – 404 / not found   → NOT_FOUND
5. classify_exception – generic error     → OTHER_ERROR
6. classify_exception – HF typed GatedRepoError (mocked)
7. classify_exception – HF typed RepositoryNotFoundError (mocked)
8. check_dataset_availability – available (mocked load_dataset)
9. check_dataset_availability – empty dataset (StopIteration)
10. check_dataset_availability – 401 error propagates correctly
11. check_dataset_availability – datasets not installed
12. check_required_datasets – returns one result per required dataset
13. check_required_datasets – custom registry respected
14. print_availability_report – smoke test (no exception)
"""

from __future__ import annotations

import importlib
import sys
import os
from io import StringIO
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load dataset_availability without pulling in torch-dependent
# code from src/data/__init__.py.
# ---------------------------------------------------------------------------

def _load_availability_module() -> ModuleType:
    """Import src.data.dataset_availability directly, bypassing __init__.py."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Provide a lightweight stub for the data sub-package so that
    # `src.data.dataset_availability` can be imported without torch.
    pkg_name = "src.data"
    if pkg_name not in sys.modules:
        stub = ModuleType(pkg_name)
        stub.__path__ = [os.path.join(repo_root, "src", "data")]
        stub.__package__ = pkg_name
        sys.modules[pkg_name] = stub

    spec = importlib.util.spec_from_file_location(
        "src.data.dataset_availability",
        os.path.join(repo_root, "src", "data", "dataset_availability.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.data.dataset_availability"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_availability_module()
DatasetCheckResult = _mod.DatasetCheckResult
DatasetStatus = _mod.DatasetStatus
REQUIRED_DATASETS = _mod.REQUIRED_DATASETS
check_dataset_availability = _mod.check_dataset_availability
check_required_datasets = _mod.check_required_datasets
classify_exception = _mod.classify_exception
print_availability_report = _mod.print_availability_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(label: str, status: DatasetStatus, message: str = "") -> DatasetCheckResult:
    return DatasetCheckResult(
        label=label,
        dataset_id=f"fake/{label}",
        split="train",
        config=None,
        status=status,
        message=message,
    )


# ---------------------------------------------------------------------------
# 1-5: classify_exception – string-based heuristics
# ---------------------------------------------------------------------------

class TestClassifyExceptionHeuristics:
    def test_401_in_message_is_auth_required(self):
        exc = Exception("HTTP Error 401 Unauthorized")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_unauthorized_keyword_is_auth_required(self):
        exc = Exception("Request failed: unauthorized access")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_403_in_message_is_auth_required(self):
        exc = Exception("403 Forbidden: you do not have permission")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_gated_keyword_is_auth_required(self):
        exc = Exception("This is a gated dataset; you must accept terms first.")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_404_in_message_is_not_found(self):
        exc = Exception("404 Client Error: Not Found for url: https://hf.co/...")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.NOT_FOUND

    def test_not_found_keyword_is_not_found(self):
        exc = Exception("Dataset 'foo/bar' not found.")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.NOT_FOUND

    def test_does_not_exist_keyword_is_not_found(self):
        exc = Exception("The repository 'foo/bar' doesn't exist.")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.NOT_FOUND

    def test_generic_error_is_other_error(self):
        exc = Exception("Some completely unrelated error occurred.")
        status, _ = classify_exception(exc)
        assert status == DatasetStatus.OTHER_ERROR

    def test_message_is_preserved(self):
        msg = "HTTP 401: your token is invalid"
        exc = Exception(msg)
        _, detail = classify_exception(exc)
        assert msg in detail


# ---------------------------------------------------------------------------
# 6-7: classify_exception – typed HF Hub exceptions (mocked)
# ---------------------------------------------------------------------------

class TestClassifyExceptionTypedHF:
    def test_gated_repo_error_via_huggingface_hub(self):
        """If huggingface_hub exposes GatedRepoError, it should be detected."""
        class FakeGatedRepoError(Exception):
            pass

        fake_module = MagicMock()
        fake_module.GatedRepoError = FakeGatedRepoError
        fake_module.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})

        exc = FakeGatedRepoError("gated")
        with patch.dict("sys.modules", {"huggingface_hub.errors": fake_module}):
            status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_repo_not_found_via_huggingface_hub(self):
        class FakeRepoNotFoundError(Exception):
            pass

        fake_module = MagicMock()
        fake_module.GatedRepoError = type("GatedRepoError", (Exception,), {})
        fake_module.RepositoryNotFoundError = FakeRepoNotFoundError

        exc = FakeRepoNotFoundError("not found")
        with patch.dict("sys.modules", {"huggingface_hub.errors": fake_module}):
            status, _ = classify_exception(exc)
        assert status == DatasetStatus.NOT_FOUND

    def test_gated_repo_error_heuristic_fallback(self):
        """Without HF Hub installed, 'gated' keyword is still caught."""
        exc = Exception("This is a gated dataset.")
        with patch.dict("sys.modules", {"huggingface_hub.errors": None, "huggingface_hub.utils": None}):
            status, _ = classify_exception(exc)
        assert status == DatasetStatus.AUTH_REQUIRED

    def test_repo_not_found_heuristic_fallback(self):
        exc = Exception("Repository not found.")
        with patch.dict("sys.modules", {"huggingface_hub.errors": None, "huggingface_hub.utils": None}):
            status, _ = classify_exception(exc)
        assert status == DatasetStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# 8-11: check_dataset_availability
# ---------------------------------------------------------------------------

class TestCheckDatasetAvailability:
    def _mock_ds(self, row: dict | None = None):
        """Return a mock streaming dataset that yields one row."""
        if row is None:
            row = {"text": "hello"}
        ds_mock = MagicMock()
        ds_mock.__iter__ = MagicMock(return_value=iter([row]))
        return ds_mock

    def test_available_with_config(self, monkeypatch):
        ds = self._mock_ds()
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = ds
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
        status, msg = check_dataset_availability("openai/gsm8k", split="train", config="main")
        assert status == DatasetStatus.AVAILABLE
        assert msg == ""

    def test_available_without_config(self, monkeypatch):
        ds = self._mock_ds()
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = ds
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
        status, msg = check_dataset_availability("openai/humaneval", split="test")
        assert status == DatasetStatus.AVAILABLE

    def test_empty_dataset_is_available(self, monkeypatch):
        ds_mock = MagicMock()
        ds_mock.__iter__ = MagicMock(return_value=iter([]))  # empty
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = ds_mock
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
        status, msg = check_dataset_availability("some/empty", split="train")
        assert status == DatasetStatus.AVAILABLE
        assert "empty" in msg

    def test_401_error_classified_as_auth_required(self, monkeypatch):
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.side_effect = Exception("401 Unauthorized")
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
        status, msg = check_dataset_availability("private/dataset", split="train")
        assert status == DatasetStatus.AUTH_REQUIRED
        assert "401" in msg

    def test_not_found_error(self, monkeypatch):
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.side_effect = Exception("404 Not Found")
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
        status, _ = check_dataset_availability("missing/dataset", split="train")
        assert status == DatasetStatus.NOT_FOUND

    def test_datasets_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        status, msg = check_dataset_availability("any/dataset", split="train")
        assert status == DatasetStatus.OTHER_ERROR
        assert msg  # non-empty message


# ---------------------------------------------------------------------------
# 12-13: check_required_datasets
# ---------------------------------------------------------------------------

class TestCheckRequiredDatasets:
    def test_returns_result_for_all_four_required_datasets(self, monkeypatch):
        """Without network, each dataset should still get a result."""
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.side_effect = Exception("connection refused")
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

        results = check_required_datasets()
        labels = {r.label for r in results}
        assert labels == {"aime", "gsm8k", "humaneval", "mmlu"}
        assert len(results) == 4

    def test_custom_registry_is_respected(self, monkeypatch):
        custom = {
            "myds": {"name": "org/myds", "split": "train", "config": None},
        }
        ds_mock = MagicMock()
        ds_mock.__iter__ = MagicMock(return_value=iter([{"x": 1}]))
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = ds_mock
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

        results = check_required_datasets(datasets=custom)
        assert len(results) == 1
        assert results[0].label == "myds"
        assert results[0].status == DatasetStatus.AVAILABLE

    def test_all_unavailable_returns_non_available_statuses(self, monkeypatch):
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.side_effect = Exception("401 Unauthorized")
        monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

        results = check_required_datasets()
        for r in results:
            assert r.status == DatasetStatus.AUTH_REQUIRED

    def test_required_datasets_registry_has_expected_keys(self):
        assert "aime" in REQUIRED_DATASETS
        assert "gsm8k" in REQUIRED_DATASETS
        assert "humaneval" in REQUIRED_DATASETS
        assert "mmlu" in REQUIRED_DATASETS


# ---------------------------------------------------------------------------
# 14: print_availability_report – smoke tests
# ---------------------------------------------------------------------------

class TestPrintAvailabilityReport:
    def test_available_dataset_shows_checkmark(self, capsys):
        results = [_make_result("gsm8k", DatasetStatus.AVAILABLE)]
        print_availability_report(results)
        captured = capsys.readouterr()
        assert "AVAILABLE" in captured.out
        assert "GSM8K" in captured.out

    def test_auth_required_shows_guidance(self, capsys):
        results = [_make_result("humaneval", DatasetStatus.AUTH_REQUIRED, "401")]
        print_availability_report(results)
        captured = capsys.readouterr()
        assert "AUTH" in captured.out
        assert "huggingface-cli login" in captured.out
        assert "HF_TOKEN" in captured.out

    def test_not_found_shows_guidance(self, capsys):
        results = [_make_result("aime", DatasetStatus.NOT_FOUND)]
        print_availability_report(results)
        captured = capsys.readouterr()
        assert "NOT-FOUND" in captured.out
        assert "huggingface.co/datasets" in captured.out

    def test_summary_counts_correct(self, capsys):
        results = [
            _make_result("gsm8k", DatasetStatus.AVAILABLE),
            _make_result("humaneval", DatasetStatus.AUTH_REQUIRED),
            _make_result("aime", DatasetStatus.NOT_FOUND),
            _make_result("mmlu", DatasetStatus.OTHER_ERROR, "timeout"),
        ]
        print_availability_report(results)
        captured = capsys.readouterr()
        assert "1 available" in captured.out
        assert "1 auth-required" in captured.out
        assert "1 not-found" in captured.out
        assert "1 other-error" in captured.out

    def test_empty_results_no_crash(self, capsys):
        print_availability_report([])
        captured = capsys.readouterr()
        assert "DATASET AVAILABILITY REPORT" in captured.out
