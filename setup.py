import setuptools as st

with open("README.md", "r") as fh:
    long_description = fh.read()

st.setup(
    name='canopy',
    version='1',
    description='A Naive Bayes model-based reinforcement learning algorithm',
    url='https://codeberg.com/brohrer/canopy',
    download_url='https://gitlab.com/brohrer/canopy/tags/',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "": [
            "README.md",
            "LICENSE",
        ],
        "canopy": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
