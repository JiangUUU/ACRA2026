from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).with_name("multi-modal_train.py")
    runpy.run_path(str(script), run_name="__main__")
