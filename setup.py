from setuptools import setup, find_packages

import versioneer

setup(
    name="emle-engine",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Lester Hedges",
    author_email="lester.hedges@gmail.com",
    packages=find_packages(include=["emle", "emle.*", "bin/*"]),
    scripts=["bin/emle-server",
             "bin/emle-stop",
             "bin/orca",
             "bin/emle-train",
             "bin/emle-analyze"],
    include_package_data=True,
    url="https://github.com/chemle/emle-engine",
    license="GPLv2",
    description="An engine for electrostatic machine learned embedding potentials",
    zip_safe=False,
)
