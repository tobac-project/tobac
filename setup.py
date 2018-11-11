from setuptools import setup

setup(name='tobac',
      version='0.8',
      description='Tracking and object-based analysis of clouds',
      url='http://github.com/climate-processes/tobac',
      author='Max Heikenfeld',
      author_email='max.heikenfeld@physics.ox.ac.uk',
      license='GNU',
      packages=['tobac'],
      install_requires=[],#['iris','numpy','netCDF4'],
      zip_safe=False)
