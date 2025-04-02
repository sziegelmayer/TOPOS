from setuptools import setup, find_packages

setup(
    name='toposv',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'numpy>=1.24',
        'nibabel>=5.0',
        'nnunetv2>=1.0.0',
        'blosc2',
        'huggingface_hub>=0.17.0',
    ],
    include_package_data=True,
    description='TOPOS: Target Organ Prediction of Scout Views for Automated CT Scan Planning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sziegelmayer/TOPOS',
    author='Sebastian Ziegelmayer, Tristan Lemke',
    author_email='sziegelmayer@tum.de, tristan.lemke@tum.de',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'topos_predict = topos.inference:main'
        ]
    },
)
