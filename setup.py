from setuptools import setup, find_packages

setup(
    name="mmos",
    version="0.1.0",
    author="qichen",
    author_email="youremail@example.com",
    description="AI记忆管理系统",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mmos",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pandas",
    ],
) 