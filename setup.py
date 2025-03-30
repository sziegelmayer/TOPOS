from setuptools import setup, find_packages

setup(
    name='topos',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch'
        # Add others if needed (e.g. numpy, pillow, etc.)
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
)
