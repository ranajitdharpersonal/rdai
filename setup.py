from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

long_description = README.read_text(
    encoding="utf-8",
)

setup(
    name="rdai",
    version="1.1.0",
    author="Ranajit Dhar",
    author_email="contact@ranajitdhar.in",
    description="One Interface. Any AI. Unbreakable Auto-Failover.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranajitdharpersonal/rdai",
    project_urls={
        "Homepage": "https://ranajitdhar.in",
        "Repository": "https://github.com/ranajitdharpersonal/rdai",
        "Issues": "https://github.com/ranajitdharpersonal/rdai/issues",
    },
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "google-genai>=1.0.0",
        "groq>=0.11.0",
        "openai>=1.54.0",
        "python-dotenv>=1.0.1",
        "PyYAML>=6.0.2",
        "questionary>=2.0.1",
        "requests>=2.32.0",
        "rich>=13.9.0",
        "typer>=0.12.5",
        "boto3>=1.34.0",
    ],
    entry_points={
        "console_scripts": [
            "rdai=rdai.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
)