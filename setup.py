from setuptools import setup, find_packages

setup(
    name='toy-auto-diff',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/toy-auto-diff',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Toy demonstrations of auto differentiations.',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
    ],
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
