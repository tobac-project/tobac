"""This file is used to configure the Sphinx build of our documentation.
The documentation on setting this up is here: https://www.sphinx-doc.org/en/master/usage/configuration.html 
"""

# This is the standard readthedocs theme.
import pydata_sphinx_theme
import sys, os

sys.path.insert(0, os.path.abspath("extensions"))

# What Sphinx extensions do we need
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "myst_parser",
    "sphinx_design",
]


html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css", "theme_overrides.css"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "restructuredtext"}
myst_enable_extensions = ["colon_fence"]


html_theme_options = {
    "logo": {
        "image_light": "images/tobac-logo-colors.png",
        "image_dark": "images/tobac-logo-colors.png",
    },
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#fontawesome-icons
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tobac-project/tobac",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
}


project = "tobac"

master_doc = "index"

# allow dropdowns
collapse_navigation = False


# Include our custom CSS (currently for special table config)
def setup(app):
    app.add_css_file("theme_overrides.css")
    app.add_css_file("custom.css")


# This should include all modules used in tobac. These are dummy imports,
# but should include both required and optional dependencies.
autodoc_mock_imports = [
    #    "numpy",
    "scipy",
    "scikit-image",
    "pandas",
    "pytables",
    "matplotlib",
    "iris",
    "cf-units",
    "xarray",
    "cartopy",
    "trackpy",
    "numba",
    "skimage",
    "sklearn",
]

sys.path.insert(0, os.path.abspath("../"))

# Napoleon settings for configuring the Napoleon extension
# See documentation here:
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
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

nbsphinx_thumbnails = {
    "examples/Basics/Idealized-Case-1_Tracking-of-a-Test-Blob-in-2D": "_static/thumbnails/Basics_Idealized-Case-1_Tracking-of-a-Test-Blob-in-2D_Thumbnail.png",
    "examples/Basics/Idealized-Case-2_Two_crossing_Blobs": "_static/thumbnails/Basics_Idealized-Case-2_Two_crossing_Blobs_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Feature-Detection_Part_1": "_static/thumbnails/Basics_Methods-and-Parameters-for-Feature-Detection_Part_1_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Feature-Detection_Part_2": "_static/thumbnails/Basics_Methods-and-Parameters-for-Feature-Detection_Part_2_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Linking": "_static/thumbnails/Basics_Methods-and-Parameters-for-Linking_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Segmentation": "_static/thumbnails/Basics_Methods-and-Parameters-for-Segmentation_Thumbnail.png",
    "examples/Example_OLR_Tracking_model/Example_OLR_Tracking_model": "_static/thumbnails/Example_OLR_Tracking_model_Thumbnail.png",
    "examples/Example_OLR_Tracking_satellite/Example_OLR_Tracking_satellite": "_static/thumbnails/Example_OLR_Tracking_satellite_Thumbnail.png",
    "examples/Example_Precip_Tracking/Example_Precip_Tracking": "_static/thumbnails/Example_Precip_Tracking_Thumbnail.png",
    "examples/Example_Track_on_Radar_Segment_on_Satellite/Example_Track_on_Radar_Segment_on_Satellite": "_static/thumbnails/Example_Track_on_Radar_Segment_on_Satellite_Thumbnail.png",
    "examples/Example_Updraft_Tracking/Example_Updraft_Tracking": "_static/thumbnails/Example_Updraft_Tracking_Thumbnail.png",
    "examples/Example_vorticity_tracking_model/Example_vorticity_tracking_model": "_static/thumbnails/Example_vorticity_tracking_model_Thumbnail.png",
}
