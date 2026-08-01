from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rdai",
    version="1.0.1",
    author="Ranajit Dhar",
    author_email="contact@ranajitdhar.in",
    description="One Interface. Any AI. Unbreakable Auto-Failover.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranajitdharpersonal/rdai", # Eta GitHub er jonno
    project_urls={
        "Website": "https://ranajitdhar.in",
        "Bug Tracker": "https://github.com/ranajitdharpersonal/rdai/issues",
    },
    packages=find_packages(),
    install_requires=[
        "typer",
        "questionary",
        "rich",
        "pyyaml",
        "google-genai",
        "openai",
        "groq",
        "boto3",
        "requests"
    ],
    entry_points={
        "console_scripts": [
            # Tumi terminal e 'rdai' likhle kon file ta run hobe, seta ekhane thik hoy.
            # Jodi tomar ashol CLI app ta rdai/cli/main.py te thake, tahole eta thik ache:
            "rdai=rdai.cli.main:app", 
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)