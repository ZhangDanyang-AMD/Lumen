import os

from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="transformer_light",
    version="0.3.0",
    description="Lightweight AMD-native quantized training framework (FP8/MXFP8/FP4) with integrated attention kernels",
    long_description=open(os.path.join(this_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "triton",
    ],
    extras_require={
        # CK attention backend (optional — falls back to Triton if absent)
        "aiter": ["amd-aiter"],
        # All optional runtime dependencies
        "all": ["amd-aiter"],
        # Developer / CI dependencies
        "dev": [
            "amd-aiter",
            "pytest",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
