"""Keep tests from talking to a real Langfuse.

``Settings()`` reads the developer's ``.env``, which holds real Langfuse keys. Empty environment
values override the file, so tracing stays off unless a test passes keys explicitly.
"""

import os

os.environ["QACOPILOT_LANGFUSE_PUBLIC_KEY"] = ""
os.environ["QACOPILOT_LANGFUSE_SECRET_KEY"] = ""
