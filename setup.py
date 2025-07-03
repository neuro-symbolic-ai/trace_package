from setuptools import setup
import os

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements():
    """Load requirements from file, separating git repos from regular packages"""
    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")

    install_requires = []
    dependency_links = []

    for req in all_reqs:
        req = req.strip()
        if req and not req.startswith('#'):
            if req.startswith('git+'):
                # Extract package name from git URL if possible
                # For git+https://github.com/nura-j/ABSynth_dataset.git
                # We can add it as a dependency link
                dependency_links.append(req)
                # You might need to add the package name to install_requires
                install_requires.append('ABSynth_dataset')  # uncomment if needed
            else:
                install_requires.append(req)

    return install_requires, dependency_links


install_requires, dependency_links = load_requirements()

setup(
    name='trace',
    version='0.1',
    packages=['trace', 'trace.hessian', 'trace.transformer', 'trace.linguistic_probes',
              'trace.intrisic_dimensions', 'trace.tokenizer', 'trace.utils', 'trace.dataloader'],
    url='',
    license='GPLv3',
    author='nura_aljaafari',
    author_email='nuraaljaafari@gmail.com',
    description='',
    install_requires=install_requires,
    dependency_links=dependency_links
)