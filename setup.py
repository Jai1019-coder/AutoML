from setuptools import setup, find_packages

setup(
    name="automl",
    version="0.1.0",
    description="CLI‑based AutoML pipeline generator (sklearn + Groq Cloud)",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer[all]",
        "pandas",
        "scikit-learn",
        "joblib",
        "requests",
        # "groq"  # optionally, if you plan to use their client
    ],
    entry_points={
        "console_scripts": [
            "automl=automl.main:app"
        ]
    },
    python_requires=">=3.7",
)
