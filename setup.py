from setuptools import setup

setup(
    name='guacomole-epigenome',
    version='0.0.1',
    packages=['avocado'],
    license='LICENSE.txt',
    description='Avocado is a package for learning a latent representation of the human epigenome.',
    install_requires=[
        "numpy >= 1.14.2",
        "pandas >= 0.23.4",
        "theano >= 1.0.1",
        "keras >= 2.0.8",
        "tqdm >= 4.24.0",
        "tensorflow >= 1.10.0",
        "numba >= 0.42.1",
    ]
)
