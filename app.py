"""
Project: Valor - Value Assessment & Lifetime Outlook via Regression
Script: app.py
Description: Root launcher delegator for Valor Streamlit Web Application.
             Delegates execution to website/app.py.
"""

import os
import sys
from pathlib import Path

# Add current directory and website directory to Python path
ROOT_DIR = Path(__file__).parent.resolve()
WEBSITE_DIR = ROOT_DIR / "website"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(WEBSITE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSITE_DIR))

# Execute the website Streamlit app
target_script = WEBSITE_DIR / "app.py"

if not target_script.exists():
    raise FileNotFoundError(f"Website application script not found at {target_script}")

with open(target_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(target_script), "exec")
    exec(code, globals())
