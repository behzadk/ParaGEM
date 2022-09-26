
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'paragem'
copyright = '2022, Behzad Karkaria'
author = 'Behzad Karkaria'
release = '0.1'

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx.ext.autosummary', 'numpydoc', 'sphinxawesome_theme', "myst_parser"]
templates_path = ['_templates']
exclude_patterns = []
# autoclass_content = 'init'
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}
add_function_parentheses = False
pygments_style = "sphinx"
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_theme_options = {'body_max_width': '125%'}
html_collapsible_definitions = False

add_module_names = False

# from sphinx.ext.autosummary.generate import AutosummaryRenderer


# def smart_fullname(fullname):
#     parts = fullname.split(".")
#     return ".".join(parts[1:])


# def fixed_init(self, app, template_dir=None):
#     AutosummaryRenderer.__old_init__(self, app, template_dir)
#     self.env.filters["smart_fullname"] = smart_fullname


# AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
# AutosummaryRenderer.__init__ = fixed_init
