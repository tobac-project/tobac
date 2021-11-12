import sphinx_rtd_theme
import sys, os

sys.path.insert(0, os.path.abspath('extensions'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.imgmath', 'sphinx.ext.ifconfig',
              'sphinx_rtd_theme',]


html_theme = "sphinx_rtd_theme"

project = u'tobac'


def setup(app):
   app.add_css_file("theme_overrides.css")

autodoc_mock_imports = ['numpy', 'scipy', 'scikit-image', 'pandas', 'pytables', 'matplotlib', 'iris',
                        'cf-units', 'xarray', 'cartopy', 'trackpy']

sys.path.insert(0, os.path.abspath("../"))