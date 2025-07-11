from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this fuction will return the list of req. 
    '''
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(
name="Project_1",
author="Dhruv Dekate",
version='0.0.1',
author_email="ddekate194@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requiremets.txt')
)