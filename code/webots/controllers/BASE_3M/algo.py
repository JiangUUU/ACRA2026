import os

# Required by deterministic CUDA matrix operations; set before PyTorch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    controller_dir = Path(__file__).resolve().parent
    webots_root = controller_dir.parents[1]
    if str(webots_root) not in sys.path:
        sys.path.insert(0, str(webots_root))
    script = controller_dir / "multi-modal_train.py"
    runpy.run_path(str(script), run_name="__main__")
