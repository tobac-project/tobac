import sphinx_rtd_theme
import sys, os

sys.path.insert(0, os.path.abspath('extensions'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.imgmath', 'sphinx.ext.ifconfig',
              'sphinx_rtd_theme','sphinx.ext.napoleon']


html_theme = "sphinx_rtd_theme"

project = u'tobac'


def setup(app):
   app.add_css_file("theme_overrides.css")

autodoc_mock_imports = ['numpy', 'scipy', 'scikit-image', 'pandas', 'pytables', 'matplotlib', 'iris',
                        'cf-units', 'xarray', 'cartopy', 'trackpy', 'numba']

sys.path.insert(0, os.path.abspath("../"))

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True