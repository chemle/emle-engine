from setuptools import setup, find_packages

import versioneer

setup(
    name="sander-mlmm",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Lester Hedges",
    author_email="lester.hedges@gmail.com",
    packages=find_packages(include=["mlmm", "mlmm.*", "bin/*"]),
    scripts=["bin/mlmm-server", "bin/mlmm-stop", "bin/orca"],
    include_package_data=True,
    url="https://github.com/lohedges/sander-mlmm",
    license="GPLv2",
    description="ML/MM wrapper for sander",
    zip_safe=False,
)
