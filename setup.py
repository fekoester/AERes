from setuptools import setup, find_packages

setup(
    name='AERes',
    version='0.1.2',
    author='Felix Köster',
    author_email='felixk@mail.saitama-u.ac.jp',
    packages=find_packages(),
    description='A Python library for dynamical systems, reservoir computing, and attention mechanisms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Rincewind1989/AERes',
    project_urls={
        'Bug Tracker': 'https://github.com/Rincewind1989/AERes/issues'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'torch',
    ],
)
