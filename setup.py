from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vec2face',
    version='0.0.1a1',
    description='Converts vectors into drawings.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jessstringham/py-vec2face',
    author='Jessica Stringham',
    author_email='mail@jessicastringham.net',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Artistic Software',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='visualization',
    package_dir={'': 'src'},
    packages=[''],
    python_requires='>=3.5',
    install_requires=['numpy', 'matplotlib'],
    package_data={
        '': ['data/*.npz'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/jessstringham/py-vec2face/issues',
        'Source': 'https://github.com/jessstringham/py-vec2face/',
        'More fun': 'https://jessicastringham.net',
    },
)
