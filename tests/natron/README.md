# Headless Natron tests

Drive the plugin without a GUI via `NatronRenderer -t <script.py>` (see the
[Natron execution docs](https://natron.readthedocs.io/en/rb-2.5/devel/natronexecution.html)).

| Script | What it checks |
|--------|----------------|
| `check_plugin.py` | The plugin is discovered in the OFX registry and a node instantiates. Prints `RESULT: PASS/FAIL`. |
| `render_passthrough.py` | A full `Read → DepthAnything3 → Write` render completes and writes output (env: `DA3_INPUT`, `DA3_OUTPUT`). |
| `make_test_image.py` | Generates a deterministic RGB fixture (no external assets). |

Always pass `OFX_PLUGIN_PATH="$HOME/Library/OFX/Plugins"` (this Natron build does not
scan the user OFX dir by default) and `--clear-openfx-cache` (force a rescan), and
redirect stdin from `/dev/null` so the interpreter exits after sourcing.

Verified locally (M2): passthrough render is byte-exact (`mean_abs_diff=0.000`,
`corr=1.000000`) at 1920×1080 in native arm64 Natron.
