"""This file is used to configure the Sphinx build of our documentation.
The documentation on setting this up is here: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# This is the standard readthedocs theme.
import pydata_sphinx_theme
import importlib
import inspect
import sys, os

sys.path.insert(0, os.path.abspath("extensions"))


# as of sphinx 8.2.0, symlinks break images in notebooks. This code fixes it.
from pathlib import Path
from sphinx.environment import BuildEnvironment

_orig_relfn2path = BuildEnvironment.relfn2path


def _relfn2path_lexical(self, filename, docname=None):
    name = str(filename)
    if not name.startswith(("/", "\\")) and not os.path.isabs(name):
        docname = docname or self.docname
        if docname:
            doc_dir = self.doc2path(docname, base=False).parent
            filename = "/" + Path(os.path.normpath(doc_dir / name)).as_posix()
    return _orig_relfn2path(self, filename, docname)


BuildEnvironment.relfn2path = _relfn2path_lexical

# What Sphinx extensions do we need
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "myst_parser",
    "sphinx_design",
    "sphinx_toolbox.code",
    "sphinxcontrib.mermaid",
    "sphinx_codeautolink",
]


html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css", "theme_overrides.css", "api_custom.css"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

html_context = {
    "api_dir": "api/generated",
}


# Generate the api documentation when building
autoclass_content = "both"

autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "description"

templates_path = ["_templates"]


html_theme_options = {
    "logo": {
        "image_light": "images/tobac-logo-colors-notext.png",
        "image_dark": "images/tobac-logo-colors-notext.png",
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


def _api_documented_modules():
    """Return the set of tobac modules listed in the API reference index.

    Parses the autosummary block of doc/api/index.rst, whose entries are
    module names relative to ``currentmodule:: tobac``.
    """
    api_index = os.path.join(os.path.dirname(__file__), "api", "index.rst")
    modules = set()
    in_autosummary = False
    with open(api_index) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(".. autosummary::"):
                in_autosummary = True
                continue
            if not in_autosummary:
                continue
            if not stripped or stripped.startswith(":"):
                continue
            if line[:1] not in (" ", "\t"):
                in_autosummary = False
                continue
            if not stripped.startswith("tobac"):
                stripped = "tobac." + stripped
            modules.add(stripped)
    return modules


API_DOCUMENTED_MODULES = _api_documented_modules()

# autosummary renders the API module pages at builder-inited, when no docname
# is set yet, so the module currently being processed is tracked explicitly
# while autosummary works (same approach as the BuildEnvironment patch above).
_scanned_module_stack = []


def _track_autosummary_modules():
    from sphinx.ext.autosummary import generate as autosummary_generate

    if getattr(
        autosummary_generate.generate_autosummary_content, "_tobac_tracked", False
    ):
        return
    orig_generate = autosummary_generate.generate_autosummary_content

    def tracked_generate(name, obj, parent, *args, **kwargs):
        _scanned_module_stack.append((name, obj))
        try:
            return orig_generate(name, obj, parent, *args, **kwargs)
        finally:
            _scanned_module_stack.pop()

    tracked_generate._tobac_tracked = True
    autosummary_generate.generate_autosummary_content = tracked_generate


_documented_module_cache = {}


def _documented_module_from_docname(app):
    """Return (name, module) of the tobac module documented on the current page.

    Only autosummary-generated API pages (under the api_dir toctree) map to a
    module; for any other page (None, None) is returned. Stub pages such as
    ``api/generated/tobac.tracking.linking_trackpy`` resolve to the module
    that contains the documented member (``tobac.tracking``).
    """
    docname = app.env.temp_data.get("docname", "")
    prefix = html_context.get("api_dir", "api/generated") + "/"
    if not docname.startswith(prefix):
        return None, None
    if docname not in _documented_module_cache:
        dotted = docname[len(prefix) :].replace("/", ".")
        parts = dotted.split(".")
        found = (None, None)
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            try:
                mod = importlib.import_module(candidate)
            except Exception:
                continue
            if inspect.ismodule(mod):
                found = (candidate, mod)
                break
        _documented_module_cache[docname] = found
    return _documented_module_cache[docname]


def _current_api_module(app):
    """Return (name, module) of the tobac module whose API page is processed.

    While autosummary generates a page the module is tracked explicitly;
    while a page is read (autodoc), it is derived from the docname.
    """
    if _scanned_module_stack:
        name, obj = _scanned_module_stack[-1]
        if inspect.ismodule(obj):
            return name, obj
    return _documented_module_from_docname(app)


def skip_member_check(app, what, name, obj, skip, options):
    """Function to skip members (autodoc-skip-member)

    Parameters
    ----------
    app
    what
    name
    obj
    skip
    options

    Returns
    -------

    """
    if hasattr(obj, "__doc__") and obj.__doc__ and ":hidden:" in obj.__doc__:
        return True

    # Skip members imported from outside the tobac package (e.g. Union or
    # Optional from typing, deepcopy from copy) so that they are not
    # rendered in the API documentation.
    obj_module = getattr(obj, "__module__", None)
    if obj_module is not None and not obj_module.startswith("tobac"):
        return True

    # On the page of a regular module, also skip private helpers (e.g.
    # tobac.tracking._calc_*) and members re-imported from another tobac
    # module that has its own API page (e.g. get_statistics is documented on
    # the tobac.utils.bulk_statistics page, so it is not linked again from
    # the tobac.feature_detection page). Re-exports in a package __init__
    # (e.g. tobac.segmentation) are part of the public API and shown in full.
    doc_name, doc_mod = _current_api_module(app)
    if doc_mod is not None and not (getattr(doc_mod, "__file__", "") or "").endswith(
        "__init__.py"
    ):
        if name.startswith("_") and not name.startswith("__"):
            return True
        if (
            obj_module is not None
            and obj_module != doc_name
            and obj_module in API_DOCUMENTED_MODULES
        ):
            return True

    return skip


# Include our custom CSS (currently for special table config)
def setup(app):
    app.add_css_file("theme_overrides.css")
    app.add_css_file("custom.css")
    _track_autosummary_modules()
    app.connect("autodoc-skip-member", skip_member_check)


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
    "cftime",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "ignore-module-all": False,
    "private-members": False,
}

sys.path.insert(0, os.path.abspath("../"))

pygments_style = "sphinx"

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

nbsphinx_execute = "never"

nbsphinx_thumbnails = {
    "examples/Basics/Idealized-Case-1_Tracking-of-a-Test-Blob-in-2D": "_static/thumbnails/Basics_Idealized-Case-1_Tracking-of-a-Test-Blob-in-2D_Thumbnail.png",
    "examples/Basics/Idealized-Case-2_Two_crossing_Blobs": "_static/thumbnails/Basics_Idealized-Case-2_Two_crossing_Blobs_Thumbnail.png",
    "examples/Basics/Idealized-Case-3_Tracking-of-a-Test-Blob-in-3D": "_static/thumbnails/Basics_Idealized-Case-3_Tracking-of-a-Test-Blob-in-3D_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Feature-Detection_Part_1": "_static/thumbnails/Basics_Methods-and-Parameters-for-Feature-Detection_Part_1_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Feature-Detection_Part_2": "_static/thumbnails/Basics_Methods-and-Parameters-for-Feature-Detection_Part_2_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Linking": "_static/thumbnails/Basics_Methods-and-Parameters-for-Linking_Thumbnail.png",
    "examples/Basics/Methods-and-Parameters-for-Segmentation": "_static/thumbnails/Basics_Methods-and-Parameters-for-Segmentation_Thumbnail.png",
    "examples/Example_OLR_Tracking_model/Example_OLR_Tracking_model": "_static/thumbnails/Example_OLR_Tracking_model_Thumbnail.png",
    "examples/Example_OLR_Tracking_model_latlon_tracking/Example_OLR_Tracking_model_latlon_tracking": "_static/thumbnails/Example_OLR_Tracking_model_Thumbnail.png",
    "examples/Example_OLR_Tracking_satellite/Example_OLR_Tracking_satellite": "_static/thumbnails/Example_OLR_Tracking_satellite_Thumbnail.png",
    "examples/Example_Precip_Tracking/Example_Precip_Tracking": "_static/thumbnails/Example_Precip_Tracking_Thumbnail.png",
    "examples/Example_Track_on_Radar_Segment_on_Satellite/Example_Track_on_Radar_Segment_on_Satellite": "_static/thumbnails/Example_Track_on_Radar_Segment_on_Satellite_Thumbnail.png",
    "examples/Example_Updraft_Tracking/Example_Updraft_Tracking": "_static/thumbnails/Example_Updraft_Tracking_Thumbnail.png",
    "examples/Example_vorticity_tracking_model/Example_vorticity_tracking_model": "_static/thumbnails/Example_vorticity_tracking_model_Thumbnail.png",
    "examples/Example_low_cloud_tracking_eurec4a/Example_low_cloud_tracking_eurec4a": "_static/thumbnails/Example_low_cloud_tracking_eurac4a_Thumbnail.png",
    "examples/Example_ICON_MCS_tracking/Example_ICON_MCS_tracking": "_static/thumbnails/Example_ICON_MCS_tracking_Thumbnail.png",
    "examples/big_data_processing/parallel_processing_tobac": "_static/thumbnails/big_data_processing_parallel_processing_tobac.svg",
    "examples/Example_Append_Tracking_Radar/Example_Append_Tracking_Radar": "_static/thumbnails/Example_Append_Tracking_Radar.png",
    "examples/Example_3D_Tracking_Radar_Data/3D_Tracking_Radar_Data": "_static/thumbnails/3D_Tracking_Radar_Data.png",
}
