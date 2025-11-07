from setuptools import setup, find_packages

setup(
    name='NeuralRecoNet',
    version='1.0',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author='Your Name',
    description='Hybrid RNN + GNN Recommender System',
)
