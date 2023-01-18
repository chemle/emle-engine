from setuptools import setup, find_packages

import versioneer

setup(
    name="sander-mlmm",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    version="0.1.0",
    author="Lester Hedges",
    author_email="lester.hedges@gmail.com",
    packages=find_packages(),
    scripts=["bin/mlmm_server", "bin/orca"],
    url="https://github.com/lohedges/sander-mlmm",
    license="GPLv2",
    description="ML/MM wrapper for sander",
    zip_safe=True,
)
