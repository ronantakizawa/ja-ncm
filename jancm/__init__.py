"""ja-NCM: Japanese-input fine-tuning of Next-Concept Models.

The vendored upstream code in ../ncm uses top-level imports (``from models...``),
so importing this package puts that directory on ``sys.path``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM_DIR = os.path.join(ROOT, "ncm")
if NCM_DIR not in sys.path:
    sys.path.insert(0, NCM_DIR)
