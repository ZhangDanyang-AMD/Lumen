import os
import sys

from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_DIR = os.path.join(this_dir, "3rdparty", "aiter")
CK_DIR = os.path.join(AITER_DIR, "3rdparty", "composable_kernel")


def _check_3rdparty():
    """Warn (but don't fail) if 3rdparty submodules are missing."""
    missing = []
    if not os.path.isdir(os.path.join(AITER_DIR, "aiter")):
        missing.append("3rdparty/aiter")
    if not os.path.isdir(CK_DIR):
        missing.append("3rdparty/aiter/3rdparty/composable_kernel (CK)")
    if missing:
        print(
            "WARNING: The following third-party submodules are not initialised:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun:\n"
            "  git submodule update --init --recursive\n",
            file=sys.stderr,
        )


_check_3rdparty()

setup(
    name="transformer_light",
    version="0.3.0",
    description="Lightweight AMD-native quantized training framework (FP8/MXFP8/FP4) with integrated attention kernels",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["3rdparty*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "triton",
    ],
    extras_require={
        "aiter": ["amd-aiter"],
        "dev": ["pytest", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
