# Headless plugin-discovery check for the DepthAnything3 OFX plugin.
#
# Run with (note OFX_PLUGIN_PATH — this Natron build does not scan
# ~/Library/OFX/Plugins by default):
#   OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins" \
#     NatronRenderer --clear-openfx-cache -t tests/natron/check_plugin.py < /dev/null
#
# Prints RESULT: PASS / FAIL for easy grepping.

import sys

PLUGIN_ID = "com.tokgan.openfx.DepthAnything3"

try:
    from NatronEngine import natron  # global core-app object
except Exception as e:  # pragma: no cover
    print("IMPORT_ERROR:", repr(e))
    natron = None


def all_plugin_ids():
    if natron is None:
        return []
    try:
        return list(natron.getPluginIDs())
    except Exception as e:
        print("GETPLUGINIDS_ERROR:", repr(e))
        return []


ids = all_plugin_ids()
found = PLUGIN_ID in ids
print("TOTAL_PLUGINS:", len(ids))
print("DISCOVERED_MATCHES:", [p for p in ids if "tokgan" in p.lower()])
print("PLUGIN_FOUND_IN_REGISTRY:", found)

node = None
try:
    node = app1.createNode(PLUGIN_ID)  # noqa: F821 (Natron global AppInstance)
except Exception as e:
    print("CREATE_EXCEPTION:", repr(e))
print("NODE_CREATED:", node is not None)

print("RESULT:", "PASS" if (found and node is not None) else "FAIL")

try:
    natron.quitApplication()
except Exception:
    sys.exit(0)
