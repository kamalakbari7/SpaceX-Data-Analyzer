
from setuptools import setup, find_packages

setup(
    name='BSSApkg',
    version='0.1.0',
    author=['Kamal Akbari', 'Taslimul Hoque'],
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'requests',
        'pyyaml'
    ],
    # Additional metadata about your package
)

