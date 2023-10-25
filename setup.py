# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages

INSTALL_DEPS = ['numpy',
                'scipy',
                'cvxpy'
               ]

TEST_DEPS = ['pytest']
DEV_DEPS = []

setup(name='optimizn',
      version='0.0.9',
      author='Rohit Pandey, Akshay Sathiya',
      author_email='rohitpandey576@gmail.com, akshay.sathiya@gmail.com',
      description='A Python library for developing customized optimization '
        + 'algorithms under generalizable paradigms.',
      packages=find_packages(exclude=['tests', 'experiments', 'Images']),
      long_description='A Python library for developing customized '
        + 'optimization algorithms under generalizable paradigms like '
        + 'simulated annealing and branch and bound. Also features '\
        + 'continuous training, so algorithms can be run multiple times '
        + 'and resume from where they left off in previous runs.',
      zip_safe=False,
      install_requires=INSTALL_DEPS,
      include_package_data=True,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      extras_require={
          'dev': DEV_DEPS,
          'test': TEST_DEPS,
      },
     )
