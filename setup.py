from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements
    #requirements=[]
    #with open(file_path) as file_obj:
     #   requirements=file.obj.readlines()
      #  requirements=[req.replace("\n","") for req in requirments]

       # if HYPEN_E_DOT in requirements:
        #    requirements.remove(HYPEN_E_DOT)
    #return requirements
    
# utils.py or wherever you define your helper functions

setup(
name='mlproject',
version='0.0.1',
author='Lakshmi',
author_email='lakshmimahadevan222@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
#install_requires=get_requirements[],
package_dir={'': 'src'},



)