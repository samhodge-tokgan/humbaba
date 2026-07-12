# Self-hosted GPU CI runners

GitHub-hosted runners have **no GPU**, so the `gpu-smoke` job in `.github/workflows/ci.yml`
(the only job that exercises real CUDA inference + cross-platform depth parity) runs on a
**self-hosted** runner labelled `cuda`. The job branches on `runner.os`, so the `cuda` label
may be served by either a Windows or a Linux box — whichever GPU host is currently up.

The model is **pre-staged** on the runner (not downloaded per run):

| OS      | Default model path                              | Override repo/runner var |
|---------|-------------------------------------------------|--------------------------|
| Windows | `C:\dev\da3-models\DA3METRIC-LARGE-dyn.onnx`     | `DA3_MODEL_WIN`          |
| Linux   | `/opt/da3-models/DA3METRIC-LARGE-dyn.onnx`       | `DA3_MODEL_LINUX`        |

The job asserts `CUDA EP available: YES`, `session created (CUDA)` (not CPU fallback), and
depth `mean` within `0.01` of `0.7026` — the cross-platform parity value (macOS CoreML
`0.702644`, Linux CUDA `0.702642`, Windows CUDA `0.702647`).

---

## Windows runner (verified: `winbox-3090`, 2× RTX 3090, CUDA 12.8, cuDNN 9)

Install as a **Windows service** so it survives the SSH/RDP session ending — an
interactively launched `run.cmd` is killed when its parent session closes.

```powershell
# In C:\actions-runner (unzip the runner package there first):
#   Register AND install the service in one shot. --runasservice installs under
#   NT AUTHORITY\NETWORK SERVICE (no password needed) and auto-starts on boot.
.\config.cmd --url https://github.com/samhodge-tokgan/openfx-onnx-depthanything3 `
  --token <REG_TOKEN> --name winbox-3090 --labels cuda,gpu --runasservice --unattended --replace
```

### Gotchas discovered (all now handled)

1. **Clock skew breaks registration.** The VM booted ~1 h behind real time; the runner's
   session token is time-sensitive and the runner stayed *offline* until synced:
   ```powershell
   w32tm /resync /force
   ```
2. **The service account has a minimal environment** — three things the interactive shell
   had but `NETWORK SERVICE` did **not**:
   - **No `pwsh` (PowerShell 7)** on PATH → the `gpu-smoke` steps use `shell: powershell`
     (Windows PowerShell 5.1, always in `System32`), *not* `shell: pwsh`. All script
     constructs used (`Out-String`, `[regex]::Match`, `[math]::Abs`, `throw`) are 5.1-safe.
   - **Execution policy `Restricted`** → GitHub's generated temp `.ps1` is refused with
     `PSSecurityException / UnauthorizedAccess`. Fix machine-wide (covers `NETWORK SERVICE`):
     ```powershell
     Set-ExecutionPolicy -Scope LocalMachine -ExecutionPolicy RemoteSigned -Force
     ```
   - **`cmake` not on any PATH scope** (installed at `C:\Program Files\CMake\bin`). Add it to
     the **Machine** PATH, then **restart the service** so it re-reads the environment:
     ```powershell
     [Environment]::SetEnvironmentVariable('Path',
       ([Environment]::GetEnvironmentVariable('Path','Machine').TrimEnd(';') + ';C:\Program Files\CMake\bin'),
       'Machine')
     Restart-Service (Get-Service 'actions.runner.*').Name -Force
     ```
3. **cuDNN 9 must live in the CUDA `bin` dir (on PATH), not just bundled** — cuDNN 9 is
   multi-DLL and its siblings resolve via the default search path. See `docs/HOST_COMPATIBILITY.md`.
4. **`dumpbin`/build tools** — the VS 2022 generator finds the toolchain via the registry, so
   `cmake -G "Visual Studio 17 2022"` works without a Developer prompt.

---

## Linux runner (Rocky Linux 8) — setup checklist

Register as a **systemd service** (`svc.sh`), which is durable by design. Sync the clock
**first** (same skew class as the Windows VM would break registration).

```bash
# 0) Sync the clock (registration token is time-sensitive)
sudo chronyd -q 'server pool.ntp.org iburst' 2>/dev/null || sudo timedatectl set-ntp true
timedatectl status | grep -E 'synchronized|Local time'

# 1) Prereqs: NVIDIA driver + CUDA 12.x runtime + cuDNN 9 on the loader path;
#    cmake >= 3.24, gcc-c++, patchelf, git. (ORT's CUDA EP needs libcuda + cudnn9.)
sudo dnf install -y cmake gcc-c++ patchelf git
nvidia-smi   # must show the GPU

# 2) Unpack the runner into ~/actions-runner, then configure with the cuda label
cd ~/actions-runner
./config.sh --url https://github.com/samhodge-tokgan/openfx-onnx-depthanything3 \
  --token <REG_TOKEN> --name rocky8-gpu --labels cuda,gpu --unattended --replace

# 3) Install + start the systemd service (survives logout/reboot)
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status

# 4) Pre-stage the model where the job expects it (or set the DA3_MODEL_LINUX var)
sudo mkdir -p /opt/da3-models
sudo cp DA3METRIC-LARGE-dyn.onnx /opt/da3-models/
```

Unlike the Windows service, the systemd unit runs as the **installing user** with a normal
login-style environment, so the PowerShell/execution-policy/PATH gotchas above do **not**
apply — the Linux `gpu-smoke` branch is plain `bash`. Just ensure `cmake`, `nvidia-smi`, and
the CUDA/cuDNN libraries are visible to that user.

### Registration/removal tokens

Generated from the repo (needs `repo` scope on the `gh` auth):

```bash
gh api -X POST repos/samhodge-tokgan/openfx-onnx-depthanything3/actions/runners/registration-token --jq .token
gh api -X POST repos/samhodge-tokgan/openfx-onnx-depthanything3/actions/runners/remove-token       --jq .token
```

> **One GPU host at a time.** The Windows and Rocky VMs share the same passed-through GPU, so
> only one is up at once. Whichever is running carries the `cuda` label and services
> `gpu-smoke`; the other shows *offline* in the runner list (harmless). To decommission a box
> for good, `./config.(cmd|sh) remove --token <REMOVE_TOKEN>` so it drops out of the list.
