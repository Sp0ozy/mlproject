from setuptools import setup, find_packages

HYPER_E_DOT="-e ."

def get_requirements(path:str) -> list[str]:
    requirements = []
    with open(path, "r") as f:
        requirements = f.readlines()
    requirements = [req.strip() for req in requirements]
    if HYPER_E_DOT in requirements:
        requirements.remove(HYPER_E_DOT)
    return [req for req in requirements if req]

setup(
    name="mlproject",
    version="0.0.1",
    author="sp0ozyy",
    author_email="imolchanov210@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
