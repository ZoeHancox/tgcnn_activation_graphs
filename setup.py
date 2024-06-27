from setuptools import setup, find_packages


with open("README.md", "r") as f:
    description = f.read()

with open("requirements.txt", "r") as f:
    install_requirements = f.read().splitlines()

setup(
    name="tgcnn_act_graph",
    version="0.3.1",
    packages=find_packages(),
    install_requires=install_requirements,
    long_description=description,
    long_description_content_type="text/markdown",
)