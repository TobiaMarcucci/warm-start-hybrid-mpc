from setuptools import setup

setup(
    name='warm_start_hmpc',
    version='0.1',
    description='Proof of concept for warm start of mixed-integer quadratic programs arising in hybrid model predictive control',
    url='https://github.com/TobiaMarcucci/warm_start_hmpc',
    author='Tobia Marcucci',
    author_email='tobiam@mit.edu',
    license='MIT',
    keywords=[
        'model predictive control',
        'mixed-integer programming',
        'hybrid systems'
        ],
    install_requires=[
        'numpy',
        'gurobipy',
        'sympy'
    ],
    zip_safe=False
    )
