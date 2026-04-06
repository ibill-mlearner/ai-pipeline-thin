"""Compatibility namespace for the ``ai-pipeline-thin`` distribution.

The installable project name uses dashes (``ai-pipeline-thin``), while Python
imports require underscores. This module gives users a predictable import path
when installing directly from git.
"""

from ai_pipeline import *  # noqa: F401,F403

