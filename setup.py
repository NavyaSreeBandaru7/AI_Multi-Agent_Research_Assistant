#!/usr/bin/env python3
"""
Setup script for AI Multi-Agent Research Assistant
Professional-grade Python package setup for distribution
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read version from __init__.py
def get_version():
    version_file = Path(__file__).parent / "src" / "ai_assistant" / "__init__.py"
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    with open(requirements_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.10.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=1.3.0",
    "twine>=4.0.0",
    "wheel>=0.41.0",
]

# Optional dependencies
extras_require = {
    "dev": dev_requirements,
    "gpu": [
        "torch[cuda]>=2.1.0",
        "tensorflow-gpu>=2.14.0",
    ],
    "api": [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    "monitoring": [
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
        "py-spy>=0.3.14",
    ],
    "production": [
        "gunicorn>=21.2.0",
        "supervisor>=4.2.5",
        "nginx-python-module>=1.0.0",
    ],
    "all": dev_requirements + [
        "torch[cuda]>=2.1.0",
        "tensorflow-gpu>=2.14.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
    ]
}

setup(
    # Basic package information
    name="ai-multi-agent-research-assistant",
    version=get_version(),
    description="Advanced AI system with NLP, Generative AI, and Multi-Agent Intelligence",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="Senior AI Engineer",
    author_email="ai.engineer@example.com",
    url="https://github.com/your-username/ai-multi-agent-research-assistant",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ai-multi-agent-research-assistant/issues",
        "Source": "https://github.com/your-username/ai-multi-agent-research-assistant",
        "Documentation": "https://ai-multi-agent-research-assistant.readthedocs.io/",
        "Changelog": "https://github.com/your-username/ai-multi-agent-research-assistant/blob/main/CHANGELOG.md",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    
    # Python version requirements
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require=extras_require,
    
    # Package data
    package_data={
        "ai_assistant": [
            "data/*.json",
            "data/*.yaml",
            "data/*.txt",
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
            "models/*.pkl",
        ],
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "ai-assistant=ai_assistant.cli:main",
            "ai-assistant-server=ai_assistant.server:main",
            "ai-assistant-train=ai_assistant.training:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
        "Environment :: Web Environment",
    ],
    
    # Keywords for discoverability
    keywords=[
        "artificial-intelligence", "machine-learning", "natural-language-processing",
        "generative-ai", "multi-agent-systems", "research-assistant",
        "nlp", "transformers", "huggingface", "streamlit",
        "chatbot", "question-answering", "text-generation",
        "sentiment-analysis", "web-scraping", "knowledge-graph"
    ],
    
    # License
    license="MIT",
    
    # Additional metadata
    platforms=["any"],
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    
    # Command class for custom commands
    cmdclass={},
)
