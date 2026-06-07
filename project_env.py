"""
Load the project ``.env`` file once at import time.

Values in ``.env`` override inherited shell / conda environment variables
(``override=True``). Application code should import this module before reading
``os.environ`` for project settings.
"""

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=True)
