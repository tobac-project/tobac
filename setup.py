from setuptools import setup

setup(name='cloudtrack',
      version='0.1',
      description='Tracking and volume identification for convective clouds',
      url='http://github.com/mheikenfeld/cloudtrack',
      author='Max Heikenfeld',
      author_email='max.heikenfeld@physics.ox.ac.uk',
      license='GNU',
      packages=['cloudtrack'],
      install_requires=[],#['iris','numpy','netCDF4'],
      zip_safe=False)
