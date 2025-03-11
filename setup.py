from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="publicia",
    version="1.3.0",
    author="Skelly0",
    description="An imperial abhuman mentat interface for Ledus Banum 77 and Imperial lore",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skelly0/Publicia",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "publicia=PubliciaV13:main",
        ],
    },
)
