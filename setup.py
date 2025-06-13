from setuptools import setup
import os
PKG_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list"""

    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [str(x).strip() for x in all_reqs]

    return install_requires

setup(
    name='trace',
    version='0.1',
    packages=['trace', 'trace.hessian', 'trace.transformer', 'trace.linguistic_probes', 'trace.intrisic_dimensions',
              'trace.tokenizer', 'trace.utils', 'trace.dataloader'],
    url='',
    license='',
    author='nura_aljaafari',
    author_email='nuraaljaafari@gmail.com',
    description='',
    install_requires=load_requirements()
)
