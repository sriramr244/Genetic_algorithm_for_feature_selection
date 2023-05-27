from setuptools import setup


setup(
    name = 'FeatureSelectionUsingGA',
    version = '1.0.0',
    description='Feature selection using genetic algorithms',
    py_modules=["fsga"],
    package_dir={'':'src'},
    install_requires=["numpy==1.22.4", "matplotlib==3.7.1"]
)