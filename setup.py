from setuptools import setup, find_packages

setup(name='recommendus',
      version='1.0',
      description='Recommendus AI',
      author='Hugo Silva',
      author_email='hugotpa@gmail.com',
      url='https://www.python.org/community/sigs/current/distutils-sig',
      setup_requires=['numpy'],
      install_requires=['Flask', 'numpy'],
      packages=find_packages(),
      )