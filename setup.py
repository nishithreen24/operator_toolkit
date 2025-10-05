from setuptools import setup, find_packages

setup(
    name="operator_toolkit",
    version="0.1.0",
    description="Library for manuevering objects in Hilbert Spaces",
    author="Nishith Reen",
    author_email="nishithreen@gmail.com",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.7",
)