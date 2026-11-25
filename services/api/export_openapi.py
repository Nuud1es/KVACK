"""Write the API's OpenAPI schema to web/openapi.json (the source of the typed TS client).

python services/api/export_openapi.py            # rewrite web/openapi.json
python services/api/export_openapi.py --check    # exit 1 if it is out of date (CI)
"""

import json
import sys
from pathlib import Path

from qa_api.main import app

TARGET = Path(__file__).resolve().parents[2] / "web" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print("web/openapi.json is out of date: run python services/api/export_openapi.py")
            sys.exit(1)
        print("web/openapi.json is up to date")
    else:
        TARGET.write_text(text, encoding="utf-8")
        print(f"wrote {TARGET}")
