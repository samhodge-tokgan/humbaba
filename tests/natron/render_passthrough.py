# Headless Read -> DepthAnything3 -> Write render, driven by env vars.
#
#   DA3_INPUT=in.png DA3_OUTPUT=out.png \
#     OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins" \
#     NatronRenderer --clear-openfx-cache -t tests/natron/render_passthrough.py < /dev/null
#
# Prints RESULT: PASS once the write completes and the output file exists.

import os
import sys

PLUGIN_ID = "com.tokgan.openfx.DepthAnything3"
inp = os.environ.get("DA3_INPUT")
out = os.environ.get("DA3_OUTPUT")

try:
    from NatronEngine import natron  # noqa: F401
except Exception as e:
    print("IMPORT_ERROR:", repr(e))

if not inp or not out:
    print("RESULT: FAIL (DA3_INPUT/DA3_OUTPUT not set)")
    sys.exit(2)

app = app1  # noqa: F821  (Natron global AppInstance)

reader = app.createReader(inp)
da3 = app.createNode(PLUGIN_ID)
writer = app.createWriter(out)

if reader is None or da3 is None or writer is None:
    print("RESULT: FAIL (node creation)",
          "reader", reader is not None, "da3", da3 is not None, "writer", writer is not None)
    sys.exit(1)

da3.connectInput(0, reader)
writer.connectInput(0, da3)

# Render frame 1 through the Write node. Try both documented signatures.
try:
    app.render(writer, 1, 1)
except Exception:
    app.render([(writer, 1, 1)])

exists = os.path.exists(out)
print("OUTPUT_EXISTS:", exists, out)
print("RESULT:", "PASS" if exists else "FAIL")

try:
    natron.quitApplication()
except Exception:
    sys.exit(0)
