from setuptools import find_packages, setup

setup(
    name="fraud_detection",
    version="0.1.0",
    description="Modular IEEE-CIS Fraud Detection training pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
