import setuptools

setuptools.setup(
    name='microns_evaluation',
    version='1.0.0',
    author='Johns Hopkins University Applied Physics Laboratory',
    description='A python package for evaluation black box Machine Learning algorithms',
    packages=setuptools.find_packages() + ['microns_evaluation.scripts'],
    package_data={'microns_evaluation': ['cwl/*.cwl', 'config/*.json']},
    entry_points={
        'console_scripts': [
            'microns-2a-run=microns_evaluation.scripts.run_2a_experiment:main',
            'microns-2a-analyze=microns_evaluation.scripts.run_2a_analysis:main',
            'microns-2b-run=microns_evaluation.scripts.run_2b_experiment:main',
            'microns-2b-analyze=microns_evaluation.scripts.run_2b_analysis:main',
        ],
    },
    install_requires=[
        'cwlref-runner',
        'boto3',
        'scipy',
        'matplotlib',
        'numpy',
        'bootstrapped',
        'statsmodels',
        'seaborn',
    ],
)
