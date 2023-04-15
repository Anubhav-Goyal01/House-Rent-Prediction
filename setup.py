from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath:str) -> List[str]:
    """
    Returns the list of required packages
    """

    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if '-e.' in requirements:
            requirements.remove('-e.')

    return requirements

setup(
    name= 'house-rent-prediction',
    version='0.0.1',
    author= 'Anubhav',
    author_email = 'anubhavgoyal101@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)