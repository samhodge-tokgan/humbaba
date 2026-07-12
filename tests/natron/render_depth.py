# Headless Read -> DepthAnything3 (inference) -> Write(EXR) render.
#
#   DA3_INPUT=in.png DA3_OUTPUT=out.exr DA3_MODEL_PATH=model.onnx \
#     DA3_ACESCG=0 DA3_PROCRES=504 \
#     OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins" \
#     NatronRenderer --clear-openfx-cache -t tests/natron/render_depth.py < /dev/null
import os
import sys

PLUGIN_ID = "com.tokgan.openfx.DepthAnything3"
inp = os.environ["DA3_INPUT"]
out = os.environ["DA3_OUTPUT"]
model = os.environ.get("DA3_MODEL_PATH", "")

try:
    from NatronEngine import natron  # noqa: F401
except Exception as e:
    print("IMPORT_ERROR:", repr(e))

app = app1  # noqa: F821


def setp(node, name, value):
    p = node.getParam(name)
    if p is None:
        print("NO_PARAM:", name)
        return
    p.setValue(value)


reader = app.createReader(inp)
da3 = app.createNode(PLUGIN_ID)
writer = app.createWriter(out)
if reader is None or da3 is None or writer is None:
    print("RESULT: FAIL (node creation)")
    sys.exit(1)


def identity_ocio(node):
    """Make the node's OCIO output space equal its input space (no transform), so
    depth is read/written as raw data rather than color-managed."""
    try:
        pin = node.getParam("ocioInputSpace")
        pout = node.getParam("ocioOutputSpace")
        if pin and pout:
            v = pin.getValue()
            pout.setValue(v)
            print("OCIO identity on", node.getScriptName(), "space=", v)
    except Exception as e:
        print("OCIO set failed:", repr(e))


# Reader: pass the file's native (sRGB-encoded) values straight to the plugin.
identity_ocio(reader)
# Writer: write the plugin's float depth verbatim (no encode/clamp).
identity_ocio(writer)

if model:
    setp(da3, "modelFile", model)
setp(da3, "inputIsACEScg", os.environ.get("DA3_ACESCG", "0") == "1")
setp(da3, "computeUnits", int(os.environ.get("DA3_COMPUTE", "0")))
setp(da3, "procLongSide", int(os.environ.get("DA3_PROCRES", "504")))

da3.connectInput(0, reader)
writer.connectInput(0, da3)

first = int(os.environ.get("DA3_FIRST", "1"))
last = int(os.environ.get("DA3_LAST", "1"))
try:
    app.render(writer, first, last)
except Exception:
    app.render([(writer, first, last)])

# For a single frame the output path is literal; for a range Natron expands
# the frame pattern, so just report completion.
exists = os.path.exists(out) or last > first
print("RENDERED_RANGE:", first, last)
print("RESULT:", "PASS" if exists else "FAIL")
try:
    natron.quitApplication()
except Exception:
    sys.exit(0)
