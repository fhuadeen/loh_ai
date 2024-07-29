# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loh_ai",
    version="0.0.1",
    author="Fhuad Balogun",
    author_email="fhuadbalogun@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "langchain==0.2.11",
        "python-dotenv==1.0.0",
        "langchain-community==0.2.10",
        "chromadb==0.5.5",
        "pypdf==4.3.1",
        "openai==1.37.1",
        "tiktoken==0.7.0",
    ],
    include_package_data=True,
    package_data={}, 
    url="https://github.com/fhuadeen/loh_ai.git",
    packages=setuptools.find_packages(),
    classifiers=(                                 # Classifiers help people find your 
        "Programming Language :: Python :: 3",    # projects. See all possible classifiers 
        "License :: OSI Approved :: License", # in https://pypi.org/classifiers/
        "Operating System :: OS Independent",   
    ),
)