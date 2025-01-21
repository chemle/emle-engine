# -*- coding: utf-8 -*-
#
# emle-engine documentation build configuration file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
from __future__ import print_function

import emle.calculator
import emle.models
import emle.train

import sys
import glob
import os

# -- General configuration -----------------------------------------------
# Add any Sphinx extension module names here, as strings.
# They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.programoutput",
    "sphinx_issues",
]

# Github repo
issues_github_path = "chemle/emle-engine"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "emle-engine"
copyright = "2023-2024"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "en"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "*_test*"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- options for mathjax
# note there is no protocol given here to avoid mixing http with https
# see: http://docs.mathjax.org/en/latest/start.html#secure-cdn-access
mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?"
    "config=TeX-AMS-MML_HTMLorMML"
)

# -- Options for HTML output ---------------------------------------------

# theme
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "font-stack": "Changa, sans-serif",
        "font-stack--monospace": "Roboto Mono, monospace",
        "color-foreground-primary": "#dddddd",  # main text and headings
        "color-foreground-secondary": "#cccccc",  # secondary text
        "color-foreground-muted": "#d0d0d0",  # muted text
        "color-foreground-border": "#923eb1",  # for content borders
        "color-background-primary": "#160f30",  # for content
        "color-background-secondary": "#201146",  # for navigation + ToC
        "color-background-hover": "#4f4fb0",  # for navigation-item hover
        "color-background-hover--transparent": "#4f4fb000",
        "color-background-border": "#403333",  # for UI borders
        "color-background-item": "#411a30",  # for "background" items (eg: copybutton)
        "color-announcement-background": "#000000dd",  # announcements
        "color-announcement-text": "#eeebee",  # announcements
        "color-admonition-title-background--note": "#FFFFFF33",  # Note background
        "color-admonition-title-background--warning": "#FF000033",  # Warning background
        "color-admonition-background": "#FFFFFF11",  # Admonition backgrounds
        "color-brand-primary": "#eeeeee",  # brand colors (sidebar titles)
        "color-brand-content": "#00dfef",  # brand colors (hyperlink color)
        "color-highlight-on-target": "#333300",  # Highlighted text background
    },
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "emle-engine"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "emle-engine"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/logo.jpg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {'**': ['sourcelink.html', 'globaltoc.html']}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "emle-engine-doc"

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": None,
    "special-members": False,
    "exclude-members": "__dict__,__weakref__",
    "private-members": False,
    "inherited-members": False,
    "show-inheritance": False,
}

autoclass_content = "both"
