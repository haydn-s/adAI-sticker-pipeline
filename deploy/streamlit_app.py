"""Streamlit entrypoint for adAI Sticker Lab (see docs/DEPLOY.md).

Streamlit Community Cloud installs ``deploy/requirements.txt`` (the file next to this
one) and runs this script from the repository root. The package is imported straight
from ``src/``, so it doesn't need to be pip-installed on the server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adai_sticker.web.streamlit_ui import main  # noqa: E402

main()
