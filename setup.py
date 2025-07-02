
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="turtlewave-hdEEG",  # Use hyphen for PyPI name
    version="2.0.0",
    author="Tancy Kao",
    author_email="tancy.kao@woolcock.org.au",
    description="High-density EEG processing for sleep event detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TancyKao/TurtleWave-hdEEG",
    packages=find_packages(include=["turtlewave_hdEEG", "turtlewave_hdEEG.*", "frontend", "frontend.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose appropriate license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "matplotlib>=3.1.0",
        "h5py>=2.10.0",
        "PyQt5>=5.12.0",
        "wonambi>=7.15"  
    ],
    entry_points={
        'console_scripts': [
            'turtlewave_gui = frontend.turtlewave_gui:main',
        ],
    },
)