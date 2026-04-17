"""
Code execution reward function.

Extracts a Python code block from the model response, runs it inside a
sandboxed subprocess, and checks whether the provided unit tests pass.
Returns 1.0 on full pass, 0.0 otherwise (partial credit is not used here
to keep the reward signal binary and unambiguous).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional

from .base import BaseReward


class CodeReward(BaseReward):
    """Rule-based reward for code generation tasks.

    The ground truth must be a Python test snippet (one or more assert
    statements or a ``check(candidate)`` function call) that exercises the
    generated function.

    Args:
        timeout: Maximum execution time in seconds per test run (default 10).
        sandbox: Execution backend.  Currently only ``"subprocess"`` is
            supported; ``"e2b"`` is reserved for future cloud sandboxing.
    """

    def __init__(
        self,
        timeout: int = 10,
        sandbox: str = "subprocess",
    ) -> None:
        self.timeout = timeout
        self.sandbox = sandbox

    def __call__(
        self, response: str, ground_truth: Optional[str] = None
    ) -> float:
        """Return 1.0 if the extracted code passes *ground_truth* tests."""
        if ground_truth is None:
            return 0.0

        code = self.extract_code(response)
        if not code:
            return 0.0

        return self._run_tests(code, ground_truth)

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    def extract_code(self, text: str) -> str:
        """Extract the first Python code block from *text*.

        Tries (in order):
        1. Fenced ```python … ``` or ``` … ``` blocks.
        2. Indented blocks that parse as valid Python.
        3. The entire *text* if it parses as Python.

        Args:
            text: Raw model output.

        Returns:
            Extracted code string (empty string if nothing parsable found).
        """
        # 1. Fenced code blocks
        fenced = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        for block in fenced:
            if self._is_valid_python(block):
                return block.strip()

        # 2. Try the entire text
        if self._is_valid_python(text):
            return text.strip()

        return ""

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def _run_tests(self, code: str, tests: str) -> float:
        """Execute *code* + *tests* in a subprocess and return 1.0 on pass."""
        full_script = textwrap.dedent(code) + "\n\n" + textwrap.dedent(tests)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(full_script)
                tmp_path = tmp.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return 1.0 if result.returncode == 0 else 0.0

        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        finally:
            import os

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """Return True if *code* parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
