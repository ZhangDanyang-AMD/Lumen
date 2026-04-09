import os
import sys

project = 'Lumen'
copyright = '2026, AMD'
author = 'Danyang Zhang'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinxcontrib.mermaid',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    'source_repository': 'https://github.com/ZhangDanyang-AMD/Lumen',
    'source_branch': 'main',
    'source_directory': 'lumen-docs/docs/source/',
    'light_css_variables': {
        'color-brand-primary': '#ed1c24',
        'color-brand-content': '#c41922',
    },
    'dark_css_variables': {
        'color-brand-primary': '#ff4654',
        'color-brand-content': '#ff4654',
    },
}

html_title = 'Lumen Documentation'
html_short_title = 'Lumen'

language = 'en'
locale_dirs = ['locale/']
gettext_compact = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}
