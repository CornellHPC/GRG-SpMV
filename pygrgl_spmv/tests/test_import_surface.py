from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_cpu_safe_import_surface_without_optional_gpu_modules():
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"cupy", "torch", "rocm"}:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import pygrgl_spmv
        import pygrgl_spmv.backends as backends
        import pygrgl_spmv.backends.mkl
        import pygrgl_spmv.backends.reference
        import pygrgl_spmv.backends.cusparse
        import pygrgl_spmv.backends.rocsparse

        print(
            json.dumps(
                {
                    "root_has_cusparse": hasattr(pygrgl_spmv, "CusparseRuntime"),
                    "backends_has_cusparse": hasattr(backends, "CusparseRuntime"),
                    "root_has_rocsparse": hasattr(pygrgl_spmv, "RocsparseRuntime"),
                    "backends_has_rocsparse": hasattr(backends, "RocsparseRuntime"),
                }
            )
        )
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload == {
        "root_has_cusparse": False,
        "backends_has_cusparse": False,
        "root_has_rocsparse": False,
        "backends_has_rocsparse": False,
    }
