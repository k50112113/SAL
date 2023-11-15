from setuptools import setup, find_packages

setup(
    name          = 'SAL',
    version       = '0.0.1',
    author        = "Shao-Chun Lee",
    packages_dir  = {"sal": "sal"},
    packages      = find_packages(exclude=["examples"], include = ["sal", "sal.*"]),
    entry_points = {
        'console_scripts': [
            'sal-run       = sal.run:main',
            'sal-sample    = sal.sample:main',
            'sal-abinitio  = sal.abinitio:main',
            'sal-backtrack = sal.backtrack:main',
            'sal-histogram = sal.histogram:main',
            'sal-md        = sal.md:main',
        ]
    }
)
