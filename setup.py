import setuptools

with open('statmod/_version.py', 'r') as fid:
    exec(fid.read())

setuptools.setup(name='statmod',
                 version=__version__,
                 author='Michael Cowan',
                 url='https://www.github.com/michael-cowan/statmod',
                 description="statistical models written in numpy",
                 packages=setuptools.find_packages(),
                 python_requires='>=3',
                 install_requires=['numpy'])
