from setuptools import setup, find_packages

setup(
    name="blackjack_opt",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "fast": ["numba"],
    },
)
