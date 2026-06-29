# =============================================================================
# FORMA 3D - Windows Setup Script (CUDA or CPU)
#
# Save as app\setup_windows.ps1, then in PowerShell:
#
#   cd path\to\app
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#
# Tested on:
#   - Windows 10 / 11 with python.org Python 3.10 or 3.11
#   - NVIDIA driver R536+ (CUDA 12.1)
#   - Bare Windows without GPU (CPU fallback)
#
# Requirements:
#   - Windows 10 1809+ or Windows 11
#   - PowerShell 5.1+ (built in) or PowerShell 7+
#   - Python 3.10 or 3.11 from python.org (NOT Microsoft Store)
#   - Internet access
#   - Either: NVIDIA GPU + driver R525+, or no GPU (CPU mode)
#
# If pytorch3d wheel index has nothing for your combo, source build needs:
#   - Visual Studio Build Tools 2019/2022 with "Desktop development with C++"
#   - (CUDA mode only) CUDA Toolkit matching your PyTorch CUDA version
#
# If MSVC is missing, use WSL2 instead:
#   wsl --install
#   then copy app\ into the WSL filesystem and run ./setup_vast.sh
#
# NOTE: this file is ASCII-only on purpose. PowerShell 5.1 does NOT read
# .ps1 files as UTF-8 by default and chokes on non-ASCII characters unless
# the file has a UTF-8 BOM. Keep it ASCII when editing.
# =============================================================================

# Use Continue so stderr from native commands (e.g. expected python probe
# failures) does not become a script-terminating NativeCommandError.
# We control flow ourselves with explicit $LASTEXITCODE checks and Die().
$ErrorActionPreference = "Continue"

# -- Helpers ------------------------------------------------------------------
function Write-Info    { param($Msg) Write-Host "-- $Msg" -ForegroundColor Cyan }
function Write-Success { param($Msg) Write-Host "  [OK]  $Msg" -ForegroundColor Green }
function Write-Warn    { param($Msg) Write-Host "  [!]   $Msg" -ForegroundColor Yellow }
function Write-Err     { param($Msg) Write-Host "  [X]   ERROR: $Msg" -ForegroundColor Red }
function Die           { param($Msg) Write-Err $Msg; exit 1 }

Write-Host ""
Write-Host "=======================================================" -ForegroundColor White
Write-Host "  FORMA 3D - Windows Setup                             " -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor White
Write-Host ""

# -- Paths --------------------------------------------------------------------
$AppDir  = $PSScriptRoot
$VenvDir = Join-Path $AppDir ".venv"
$CkptDir = Join-Path $AppDir "GVHMR\inputs\checkpoints"

# =============================================================================
# STEP 0 - Preflight checks
# =============================================================================
Write-Info "Step 0 - Preflight checks"

# Windows version
$WinVer = [System.Environment]::OSVersion.Version
if ($WinVer.Major -lt 10) {
    Die "Windows 10 or later required (detected: $($WinVer))"
}
Write-Success "Windows $($WinVer.Major).$($WinVer.Build)"

# PowerShell version
$PsVer = $PSVersionTable.PSVersion
if ($PsVer.Major -lt 5) {
    Die "PowerShell 5.1+ required (detected: $PsVer)"
}
Write-Success "PowerShell $PsVer"

# Architecture: x64 only
$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Arch -ne "AMD64") {
    Write-Warn "Architecture: $Arch - only AMD64 (x64) is tested. ARM64 will likely fail at pytorch3d."
} else {
    Write-Success "Architecture: x64"
}

# Long-path support (260-char MAX_PATH bites pip on deep nested deps)
$LongPaths = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                              -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
if (-not $LongPaths -or $LongPaths.LongPathsEnabled -ne 1) {
    Write-Warn "Long paths (>260 chars) not enabled in the registry."
    Write-Warn "  pip may fail on deep pytorch3d / lightning paths."
    Write-Warn "  Fix (in admin PowerShell):"
    Write-Warn "    New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' ``"
    Write-Warn "      -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
    Write-Warn "  Or place app\ somewhere short like C:\forma\app to avoid the issue."
} else {
    Write-Success "Long-path support enabled"
}

# Python: prefer py launcher 3.11 then 3.10, fall back to python
$Python  = $null
$PyVer   = $null
$PyMinor = $null

foreach ($candidate in @(
    @{Cmd = "py";     Probe = @("-3.11", "--version")},
    @{Cmd = "py";     Probe = @("-3.10", "--version")},
    @{Cmd = "python"; Probe = @("--version")}
)) {
    $verOut = & $candidate.Cmd $candidate.Probe 2>$null
    if ($LASTEXITCODE -eq 0 -and $verOut -match "Python 3\.(\d+)") {
        $minor = [int]$Matches[1]
        if ($minor -ge 10 -and $minor -lt 12) {
            $launchArgs = $candidate.Probe | Select-Object -First ($candidate.Probe.Count - 1)
            $Python  = @{Cmd = $candidate.Cmd; Args = $launchArgs}
            $PyVer   = "3.$minor"
            $PyMinor = $minor
            break
        }
    }
}

if (-not $Python) {
    Die @"
Python 3.10 or 3.11 not found.

Install from https://www.python.org/downloads/windows/  (NOT Microsoft Store).

During install, tick:
  [x] Add python.exe to PATH
  [x] tcl/tk and IDLE  (required: GVHMR imports tkinter)

Or via winget (admin PowerShell):
  winget install Python.Python.3.11
"@
}

Write-Success "Python: $($Python.Cmd) $($Python.Args -join ' ') ($PyVer)"

# Detect Microsoft Store Python (sandboxed - breaks pytorch3d / ONNX Runtime)
$PyExePath = & $Python.Cmd $Python.Args -c "import sys; print(sys.executable)" 2>$null
if ($PyExePath -like "*WindowsApps*") {
    Write-Warn "You're using Microsoft Store Python ($PyExePath)."
    Write-Warn "  Its sandboxed site-packages will break pytorch3d and ONNX Runtime."
    Write-Warn "  Strongly recommended: uninstall, reinstall from python.org."
}

# tkinter
& $Python.Cmd $Python.Args -c "import tkinter" *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "tkinter not importable from Python."
    Write-Warn "  GVHMR will fail. Reinstall Python with 'tcl/tk and IDLE' ticked."
} else {
    Write-Success "tkinter importable"
}

# git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Warn "git not found - required for pytorch3d source-build fallback."
    Write-Warn "  Install: winget install Git.Git  (then reopen this PowerShell)"
} else {
    $GitVer = git --version
    Write-Success "git: $GitVer"
}

# MSVC (cl.exe) - required for pytorch3d source build on Windows.
# Even when VS Build Tools is installed, cl.exe is NOT on PATH unless the
# session was launched from a Developer Shell or vcvars was sourced. We try
# to find it via vswhere and load the VC environment into THIS process so
# downstream pip builds (Step 4) inherit a working compiler, instead of
# discovering the problem 15 minutes into a doomed build.
$HasCl = $false

if (Get-Command cl -ErrorAction SilentlyContinue) {
    $HasCl = $true
    # Same DISTUTILS_USE_SDK requirement applies here: if cl.exe is already
    # on PATH, the VC environment is already active (e.g. launched from a
    # Developer PowerShell shortcut), and torch's ABI check needs to know
    # that explicitly rather than guessing.
    $env:DISTUTILS_USE_SDK = "1"
    Write-Success "MSVC: cl.exe already on PATH"
} else {
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VsInstallPath = $null

    if (Test-Path $VsWhere) {
        $VsInstallPath = & $VsWhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
    }

    if ($VsInstallPath) {
        $VcVarsAll = Join-Path $VsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path $VcVarsAll) {
            Write-Host "  Found VS Build Tools at $VsInstallPath - loading x64 VC environment ..."

            # cmd.exe /c runs vcvarsall then dumps the resulting env so we can
            # import it into this PowerShell session (vcvarsall itself only
            # affects the cmd.exe child process, not us, unless we capture it).
            $EnvDump = cmd.exe /c "`"$VcVarsAll`" x64 >nul 2>&1 && set" 2>$null
            foreach ($line in $EnvDump) {
                if ($line -match '^([^=]+)=(.*)$') {
                    Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] -ErrorAction SilentlyContinue
                }
            }

            if (Get-Command cl -ErrorAction SilentlyContinue) {
                $HasCl = $true
                # torch's cpp_extension._check_abi() hard-fails if VC env vars
                # are active but DISTUTILS_USE_SDK isn't set - it can't tell
                # whether the VC env was already loaded (as we just did) or
                # needs activating itself, and refuses to risk a double
                # activation. Setting this tells it "already handled."
                $env:DISTUTILS_USE_SDK = "1"
                Write-Success "MSVC: cl.exe loaded from VS Build Tools (this session only)"
            }
        }
    }

    if (-not $HasCl) {
        Write-Warn "MSVC (cl.exe) not found - pytorch3d source build (Step 4) WILL fail."
        Write-Warn "  Install Visual Studio Build Tools 2022:"
        Write-Warn "    winget install Microsoft.VisualStudio.2022.BuildTools"
        Write-Warn "  During install, tick 'Desktop development with C++'."
        Write-Warn "  Then reopen PowerShell and re-run this script."
        Write-Warn ""
        Write-Warn "  Alternative: skip this entirely with WSL2 (recommended, no MSVC needed):"
        Write-Warn "    wsl --install -d Ubuntu"
        Write-Warn "    Then copy app\ into the WSL filesystem and run ./setup_vast.sh"
    }
}

# ffmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warn "ffmpeg not on PATH - MP4 export will fall back to imageio-ffmpeg's bundled binary (still works)."
    Write-Warn "  For full ffmpeg features: winget install Gyan.FFmpeg  (then reopen PowerShell)"
} else {
    $FfVer = (ffmpeg -version 2>$null | Select-Object -First 1)
    Write-Success "ffmpeg: $FfVer"
}

# =============================================================================
# STEP 0b - GPU / CUDA detection
# =============================================================================
Write-Info "Step 0b - GPU / CUDA detection"

$GpuMode = "cpu"
$CudaTag = ""
$CudaVer = ""

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $NvOut = nvidia-smi 2>$null | Out-String
    if ($NvOut -match "CUDA Version:\s*(\d+\.\d+)") {
        $CudaVer = $Matches[1]
        $GpuMode = "cuda"

        $parts = $CudaVer.Split(".")
        $cm = [int]$parts[0]
        $cn = [int]$parts[1]

        if     ($cm -ge 12 -and $cn -ge 4) { $CudaTag = "cu124" }
        elseif ($cm -ge 12)                { $CudaTag = "cu121" }
        elseif ($cm -ge 11 -and $cn -ge 8) { $CudaTag = "cu118" }
        else {
            Write-Warn "Detected CUDA $CudaVer < 11.8 - forcing cu118. Update your driver if PyTorch fails to init."
            $CudaTag = "cu118"
        }
        Write-Success "GPU detected - CUDA $CudaVer, using wheels: $CudaTag"

        $GpuLine = ($NvOut -split "`n") | Where-Object { $_ -match "\d+%\s*Default" } | Select-Object -First 1
        if ($GpuLine) { Write-Success "GPU: $($GpuLine.Trim())" }
    } else {
        Write-Warn "nvidia-smi exists but didn't report a CUDA version - falling back to CPU"
    }
} else {
    Write-Warn "No nvidia-smi on PATH - installing CPU-only PyTorch (much slower)"
}

# =============================================================================
# STEP 1 - Virtual environment
# =============================================================================
Write-Info "Step 1 - Virtual environment"

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvDir) {
    $venvOK = $false
    if (Test-Path $VenvPython) {
        & $VenvPython --version *>$null
        if ($LASTEXITCODE -eq 0) { $venvOK = $true }
    }

    if ($venvOK) {
        Write-Success "Virtual environment already exists - reusing"
    } else {
        Write-Warn ".venv exists but python.exe isn't runnable (probably from another OS). Rebuilding ..."
        Remove-Item -Recurse -Force $VenvDir
        & $Python.Cmd $Python.Args -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }
        Write-Success "Virtual environment rebuilt"
    }
} else {
    Write-Host "  Creating .venv inside app\ ..."
    & $Python.Cmd $Python.Args -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }
    Write-Success "Virtual environment created"
}

$Pip     = Join-Path $VenvDir "Scripts\pip.exe"
$Python3 = Join-Path $VenvDir "Scripts\python.exe"

# pip self-upgrade has to go via "python -m pip" on Windows.
# Calling pip.exe directly to upgrade itself fails with:
#   "ERROR: To modify pip, please run the following command: ... -m pip ..."
& $Python3 -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip self-upgrade reported a non-zero exit code (often benign). Continuing ..."
} else {
    Write-Success "pip / setuptools / wheel upgraded"
}

& $Python3 -c "import _tkinter" *>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "tkinter (_tkinter) importable inside venv"
} else {
    Write-Warn "_tkinter not importable. Reinstall Python with 'tcl/tk and IDLE' option ticked."
}

# =============================================================================
# STEP 2 - PyTorch (CUDA or CPU)
# =============================================================================
Write-Info "Step 2 - PyTorch ($GpuMode)"

# Probe wraps `import torch` in try/except so the import failure (when torch
# isn't installed yet) exits with code 1 cleanly, without dumping a traceback
# to stderr that would otherwise cause PowerShell drama.
$TorchProbe = @"
import sys
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    ok = (major, minor) >= (2, 1)
    if ok and '$GpuMode' == 'cuda':
        ok = torch.cuda.is_available()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
"@

& $Python3 -c $TorchProbe *>$null
$NeedTorch = ($LASTEXITCODE -ne 0)

if (-not $NeedTorch) {
    $TorchVer = & $Python3 -c "import torch; print(torch.__version__)"
    $CudaOK   = & $Python3 -c "import torch; print(torch.cuda.is_available())"
    Write-Success "PyTorch $TorchVer already installed (cuda.is_available=$CudaOK) - skipping"
} else {
    if ($GpuMode -eq "cuda") {
        Write-Host "  Installing PyTorch + torchvision with $CudaTag wheels (~2.5 GB, please wait) ..."
        & $Pip install --quiet torch torchvision `
            --index-url "https://download.pytorch.org/whl/$CudaTag"
    } else {
        Write-Host "  Installing PyTorch + torchvision (CPU wheels, ~200 MB) ..."
        & $Pip install --quiet torch torchvision `
            --index-url "https://download.pytorch.org/whl/cpu"
    }
    if ($LASTEXITCODE -ne 0) { Die "PyTorch install failed" }
    Write-Success "PyTorch installed"
}

$TorchReport = @"
import sys
try:
    import torch
    print(f'  torch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA available - {torch.cuda.get_device_name(0)}')
    else:
        print('  No GPU - CPU mode')
except Exception as e:
    print(f'  torch report failed: {e}')
    sys.exit(1)
"@
& $Python3 -c $TorchReport
if ($LASTEXITCODE -ne 0) { Die "torch import failed after install - check the pip output above" }

# =============================================================================
# STEP 3 - All pip dependencies
# =============================================================================
Write-Info "Step 3 - pip dependencies"

# 1. numpy first, pinned to 1.26.4. onnxruntime is built against numpy 1.x ABI;
#    2.x segfaults inside onnxruntime_pybind11_state.pyd.
Write-Host "  [1/10] numpy, scipy, shapely, pyyaml, easydict, matplotlib ..."
& $Pip install "numpy==1.26.4" scipy shapely pyyaml easydict matplotlib --quiet
if ($LASTEXITCODE -ne 0) { Die "numpy / scipy / shapely / pyyaml install failed" }

# 2. FastAPI server
Write-Host "  [2/10] fastapi, uvicorn, python-multipart ..."
& $Pip install fastapi "uvicorn[standard]" python-multipart --quiet
if ($LASTEXITCODE -ne 0) { Die "fastapi / uvicorn install failed" }

# 3. OpenCV (headless - no Qt/GUI deps)
Write-Host "  [3/10] opencv-python-headless ..."
& $Pip install opencv-python-headless --quiet
if ($LASTEXITCODE -ne 0) { Die "opencv-python-headless install failed" }

# 4. Video I/O
Write-Host "  [4/10] imageio, imageio-ffmpeg, av ..."
& $Pip install "imageio[ffmpeg]" imageio-ffmpeg --quiet
if ($LASTEXITCODE -ne 0) { Die "imageio install failed" }
& $Pip install av --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "av (PyAV) failed - imageio fallback reader will be used (OK)"
}

# 5. ffmpeg-python
Write-Host "  [5/10] ffmpeg-python ..."
& $Pip install ffmpeg-python --quiet
if ($LASTEXITCODE -ne 0) { Die "ffmpeg-python install failed" }

# 6. YOLO + BoT-SORT
Write-Host "  [6/10] ultralytics ..."
& $Pip install ultralytics --quiet
if ($LASTEXITCODE -ne 0) { Die "ultralytics install failed" }

# 7. Team clustering (ResNet-26 embeddings)
Write-Host "  [7/10] transformers ..."
& $Pip install transformers --quiet
if ($LASTEXITCODE -ne 0) { Die "transformers install failed" }

# 8. ONNX Runtime - GPU build when CUDA available
Write-Host "  [8/10] onnxruntime ..."
if ($GpuMode -eq "cuda") {
    & $Pip install onnxruntime-gpu --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "onnxruntime-gpu failed - falling back to onnxruntime (CPU)"
        & $Pip install onnxruntime --quiet
        if ($LASTEXITCODE -ne 0) { Die "onnxruntime install failed" }
    }
} else {
    & $Pip install onnxruntime --quiet
    if ($LASTEXITCODE -ne 0) { Die "onnxruntime install failed" }
}
& $Pip install "numpy==1.26.4" --force-reinstall --quiet
if ($LASTEXITCODE -ne 0) { Die "numpy 1.26.4 force-reinstall failed" }

# 9. GVHMR / SMPL-X stack
Write-Host "  [9/10] GVHMR stack: colorlog, einops, hydra, lightning, timm, smplx, wis3d ..."
& $Pip install colorlog einops --quiet
if ($LASTEXITCODE -ne 0) { Die "colorlog/einops install failed" }
& $Pip install "hydra-core>=1.3" omegaconf --quiet
if ($LASTEXITCODE -ne 0) { Die "hydra-core/omegaconf install failed" }
& $Pip install hydra-zen --quiet
if ($LASTEXITCODE -ne 0) { Die "hydra-zen install failed" }
& $Pip install "lightning>=2.0" pytorch-lightning --quiet
if ($LASTEXITCODE -ne 0) { Die "lightning install failed" }
& $Pip install "timm>=0.9" --quiet
if ($LASTEXITCODE -ne 0) { Die "timm install failed" }
& $Pip install --no-build-isolation smplx chumpy --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "smplx + chumpy combined install failed - trying smplx alone (chumpy is optional)"
    & $Pip install --no-build-isolation smplx --quiet
    if ($LASTEXITCODE -ne 0) { Die "smplx install failed" }
}
& $Pip install wis3d --quiet
if ($LASTEXITCODE -ne 0) { Die "wis3d install failed" }

# 10. pytorch3d build prerequisites
Write-Host "  [10/10] fvcore, iopath, ninja ..."
& $Pip install fvcore iopath ninja --quiet
if ($LASTEXITCODE -ne 0) { Die "fvcore/iopath/ninja install failed" }

Write-Success "All pip dependencies installed"

# =============================================================================
# STEP 4 - pytorch3d (community wheel preferred, source build fallback)
# =============================================================================
Write-Info "Step 4 - pytorch3d (the painful one on Windows)"

# 'import pytorch3d' alone is NOT a reliable health check: pytorch3d/__init__.py
# does not eagerly import the native _C extension, so a build/wheel with a
# broken _C.pyd (wrong ABI, missing CUDA DLLs on a CPU box, etc.) will still
# pass a plain import - the failure only surfaces later when something calls
# into pytorch3d.ops. We force that exercise here so a bad install is caught
# and rebuilt now, not mid-pipeline in lift_gvhmr.py.
$P3dHealthCheck = "import pytorch3d; import pytorch3d.ops.knn"

& $Python3 -c $P3dHealthCheck *>$null
if ($LASTEXITCODE -eq 0) {
    $P3dVer = & $Python3 -c "import pytorch3d; print(pytorch3d.__version__)"
    Write-Success "pytorch3d $P3dVer already installed and _C loads OK - skipping"
} else {
    $WheelOK = $false

    # MiroPsota's community wheel index supports BOTH CUDA and CPU on Windows.
    # NOTE: this index has been a source of installs that import fine but
    # ship a _C.pyd that fails to load at runtime on this machine (confirmed
    # by the user - source build from git works, the community wheel does
    # not). We still try it first since it's much faster when it works, but
    # we verify with the real _C health check below rather than trusting
    # a plain import, so a bad wheel correctly falls through to source build
    # instead of being accepted as a false success.
    $TorchVerStr = & $Python3 -c "import torch; print(torch.__version__.split('+')[0])"
    if ($GpuMode -eq "cuda") { $WheelTag = "${TorchVerStr}${CudaTag}" }
    else                     { $WheelTag = "${TorchVerStr}cpu" }
    $CpTag = "cp${PyMinor}0"   # cp310 / cp311

    Write-Host "  Trying community wheel index (MiroPsota) ..."
    Write-Host "  Looking for: pt$WheelTag / $CpTag / win_amd64"

    $WheelUrl = "https://miropsota.github.io/torch_packages_builder/pytorch3d/"
    & $Pip install --quiet --no-cache-dir pytorch3d -f $WheelUrl
    $PipExit = $LASTEXITCODE

    if ($PipExit -eq 0) {
        & $Python3 -c $P3dHealthCheck *>$null
        if ($LASTEXITCODE -eq 0) {
            $P3dVer = & $Python3 -c "import pytorch3d; print(pytorch3d.__version__)"
            Write-Success "pytorch3d $P3dVer installed from community wheel"
            $WheelOK = $true
        }
    }

    if (-not $WheelOK) {
        if ($PipExit -eq 0) {
            Write-Warn "Community wheel installed but its _C extension failed to load - uninstalling broken wheel before source build."
            & $Pip uninstall pytorch3d -y --quiet *>$null
        } else {
            Write-Warn "No matching community wheel for this combo. Falling back to source build."
        }
        Write-Host ""

        if (-not $HasCl) {
            Die @"
pytorch3d source build cannot proceed: MSVC (cl.exe) was not found earlier
in Step 0 (see warning above), and there is no prebuilt wheel for your
torch+Python combo. Without a C++ compiler this build cannot succeed -
skipping the attempt instead of wasting 10-20 minutes on a guaranteed failure.

Recovery options:

  1. Install Visual Studio Build Tools 2022, then re-run this script:
       winget install Microsoft.VisualStudio.2022.BuildTools
       (during install pick 'Desktop development with C++')
     Then CLOSE and REOPEN PowerShell (env vars only load in new shells)
     and re-run this script - Step 0 will auto-detect and load cl.exe.

  2. Use WSL2 instead (recommended - no MSVC needed, ~8 min on CPU):
       wsl --install -d Ubuntu
       wsl
     Then copy app\ into the WSL filesystem and run ./setup_vast.sh

  3. Wait for a community wheel matching your torch+Python combo at:
       https://miropsota.github.io/torch_packages_builder/pytorch3d/
"@
        }

        if ($GpuMode -eq "cuda") {
            Write-Warn "pytorch3d source build on Windows + CUDA needs:"
            Write-Warn "  - Visual Studio Build Tools 2022 + 'Desktop development with C++' (found OK)"
            Write-Warn "  - CUDA Toolkit $CudaVer (toolkit, not just driver)"
            Write-Warn "  - 20-40 min compile time"
        } else {
            Write-Warn "pytorch3d source build on Windows + CPU needs:"
            Write-Warn "  - Visual Studio Build Tools 2022 + 'Desktop development with C++' (found OK)"
            Write-Warn "  - 10-20 min compile time"
        }
        Write-Host ""

        # FORCE_CUDA is set unconditionally (not just inside an if/else keyed
        # on $GpuMode) so a CPU build can never silently inherit CUDA kernel
        # compilation from a stray env var or a GpuMode detection edge case.
        # Without this pinned to "0", setup.py can probe for a CUDA toolkit
        # and compile CUDA code into _C.pyd even when torch itself is CPU-only,
        # producing a _C.pyd that fails to load (missing CUDA runtime DLLs)
        # rather than failing to build.
        $env:MAX_JOBS = "4"
        $env:FORCE_CUDA = "0"
        if ($GpuMode -eq "cuda") {
            $env:FORCE_CUDA = "1"
            if (-not $env:CUDA_HOME -and (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")) {
                $LatestCuda = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" |
                              Sort-Object Name -Descending | Select-Object -First 1
                if ($LatestCuda) {
                    $env:CUDA_HOME = $LatestCuda.FullName
                    Write-Host "  Using CUDA_HOME = $env:CUDA_HOME"
                }
            }
        }
        Write-Host "  FORCE_CUDA = $env:FORCE_CUDA"

        # ninja was pip-installed in Step 3, but pip-installed console scripts
        # land in .venv\Scripts - confirm it's actually invocable from here,
        # not just present in site-packages, before sinking 10-20 min into a
        # build that silently falls back to slow non-ninja compilation
        # (or fails outright) if it isn't.
        & $Python3 -m ninja --version *>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "ninja not invocable via '$Python3 -m ninja' - reinstalling ..."
            & $Pip install --force-reinstall ninja --quiet
        }
        $NinjaVer = & $Python3 -m ninja --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "ninja $NinjaVer ready (build will use parallel compilation)"
        } else {
            Write-Warn "ninja still not invocable - build will fall back to slower distutils compilation"
        }

        & $Pip install --no-build-isolation `
            "git+https://github.com/facebookresearch/pytorch3d.git"
        $BuildExit = $LASTEXITCODE

        & $Python3 -c $P3dHealthCheck *>$null
        if ($BuildExit -eq 0 -and $LASTEXITCODE -eq 0) {
            $P3dVer = & $Python3 -c "import pytorch3d; print(pytorch3d.__version__)"
            Write-Success "pytorch3d $P3dVer built and installed, _C loads OK"
        } else {
            Die @"
pytorch3d source build failed on Windows.

Recovery options:

  1. Use WSL2 (recommended). From admin PowerShell:
       wsl --install -d Ubuntu
       wsl
     Then copy app\ into the WSL filesystem and run ./setup_vast.sh - that
     script does NOT have this problem and finishes in ~8 minutes on CPU.

  2. Install the missing build tools:
       winget install Microsoft.VisualStudio.2022.BuildTools
       (during install pick 'Desktop development with C++')
     Then reopen PowerShell and re-run this script.

  3. Wait for a community wheel matching your torch+Python combo at:
       https://miropsota.github.io/torch_packages_builder/pytorch3d/
"@
        }
    }
}

# =============================================================================
# STEP 5 - Import smoke test
# =============================================================================
Write-Info "Step 5 - Import smoke test"

$SmokeScript = @"
import sys, importlib

REQUIRED = [
    ('fastapi',           'fastapi'),
    ('uvicorn',           'uvicorn[standard]'),
    ('multipart',         'python-multipart'),
    ('numpy',             'numpy'),
    ('cv2',               'opencv-python-headless'),
    ('scipy',             'scipy'),
    ('yaml',              'pyyaml'),
    ('easydict',          'easydict'),
    ('shapely',           'shapely'),
    ('matplotlib',        'matplotlib'),
    ('ultralytics',       'ultralytics'),
    ('transformers',      'transformers'),
    ('torch',             'torch'),
    ('torchvision',       'torchvision'),
    ('onnxruntime',       'onnxruntime / onnxruntime-gpu'),
    ('imageio',           'imageio[ffmpeg]'),
    ('imageio_ffmpeg',    'imageio-ffmpeg'),
    ('ffmpeg',            'ffmpeg-python'),
    ('smplx',             'smplx'),
    ('einops',            'einops'),
    ('hydra',             'hydra-core'),
    ('omegaconf',         'omegaconf'),
    ('hydra_zen',         'hydra-zen'),
    ('lightning',         'lightning'),
    ('pytorch_lightning', 'pytorch-lightning'),
    ('timm',              'timm'),
    ('colorlog',          'colorlog'),
    ('wis3d',             'wis3d'),
    ('fvcore',            'fvcore'),
    ('iopath',            'iopath'),
]

OPTIONAL = [
    ('tkinter',  'tkinter (reinstall Python with tcl/tk option)'),
    ('av',       'av (PyAV)'),
    ('chumpy',   'chumpy'),
]

failed = []
for mod, pip_name in REQUIRED:
    try:
        importlib.import_module(mod)
        print(f'  OK   {mod}')
    except ImportError as exc:
        print(f'  FAIL {mod}  ->  pip install {pip_name}  ({exc})')
        failed.append(pip_name)

# pytorch3d gets a dedicated check, not the generic loop above: its
# __init__.py does not eagerly import the native _C extension, so a plain
# 'import pytorch3d' silently passes even when _C.pyd fails to load (wrong
# ABI, missing CUDA DLLs on a CPU box, etc.). We only find out for sure by
# importing a submodule that actually touches _C, same as GVHMR's real
# import path (pytorch3d.ops.knn) does at runtime.
try:
    import pytorch3d
    import pytorch3d.ops.knn
    print(f'  OK   pytorch3d {pytorch3d.__version__} (_C loads OK)')
except ImportError as exc:
    print(f'  FAIL pytorch3d  ->  _C extension failed to load ({exc})')
    print('       This means the install "succeeded" but the native build is broken.')
    print('       Try: pip uninstall pytorch3d -y && pip install --no-build-isolation \\')
    print('              "git+https://github.com/facebookresearch/pytorch3d.git"')
    failed.append('pytorch3d (broken _C - see above)')

for mod, pip_name in OPTIONAL:
    try:
        importlib.import_module(mod)
        print(f'  OK   {mod}  (optional)')
    except ImportError:
        print(f'  WARN {mod}  (optional - not installed, OK)')

if failed:
    print('\nFailed: ' + ', '.join(failed))
    sys.exit(1)

print('\n  All required imports OK')

try:
    import torch, onnxruntime as ort
    if torch.cuda.is_available():
        provs = ort.get_available_providers()
        cuda_ok = 'CUDAExecutionProvider' in provs
        flag = 'yes' if cuda_ok else 'NO - install onnxruntime-gpu'
        print(f'\n  CUDA ready: torch={torch.cuda.get_device_name(0)}, ORT CUDAExecutionProvider={flag}')
    else:
        print('\n  Running in CPU mode')
except Exception as e:
    print(f'\n  runtime check skipped: {e}')
"@

& $Python3 -c $SmokeScript
if ($LASTEXITCODE -ne 0) {
    Die "Smoke test failed. See errors above."
}
Write-Success "Smoke test passed"

# =============================================================================
# STEP 6 - Model file check
# =============================================================================
Write-Info "Step 6 - Model file check"

$Missing = 0
function Check-File {
    param($Path, $Label)
    if (Test-Path $Path) {
        $Size = "{0:N1} MB" -f ((Get-Item $Path).Length / 1MB)
        Write-Success "$Label  ($Size)"
    } else {
        Write-Err "MISSING: $Label"
        Write-Host "         Expected at: $Path"
        $script:Missing += 1
    }
}

Write-Host "  models\:"
Check-File (Join-Path $AppDir "models\player_detection_v26s.pt") "player_detection_v26s.pt"
Check-File (Join-Path $AppDir "models\vitpose-b-coco.onnx")      "vitpose-b-coco.onnx"
Check-File (Join-Path $AppDir "models\SV_kp")                    "SV_kp  (PnLCalib)"
Check-File (Join-Path $AppDir "models\SV_lines")                 "SV_lines  (PnLCalib)"

Write-Host ""
Write-Host "  GVHMR\inputs\checkpoints\:"
Check-File (Join-Path $CkptDir "gvhmr\gvhmr_siga24_release.ckpt")     "gvhmr_siga24_release.ckpt"
Check-File (Join-Path $CkptDir "hmr2\epoch=10-step=25000.ckpt")       "hmr2 epoch=10-step=25000.ckpt"
Check-File (Join-Path $CkptDir "body_models\smplx\SMPLX_NEUTRAL.npz") "SMPLX_NEUTRAL.npz"
Check-File (Join-Path $CkptDir "vitpose\vitpose-h-multi-coco.pth")    "vitpose-h-multi-coco.pth"
Check-File (Join-Path $CkptDir "yolo\yolov8x.pt")                     "yolov8x.pt"

if ($Missing -gt 0) {
    Write-Warn "$Missing model file(s) missing - affected pipeline stages will fail at runtime."
    Write-Warn "Run  python download_models.py  to fetch the public ones."
    Write-Warn "GVHMR weights must be uploaded manually (license)."
}

# =============================================================================
# STEP 7 - Runtime directories
# =============================================================================
Write-Info "Step 7 - Runtime directories"
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "uploads")           | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "outputs")           | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "outputs\transcode") | Out-Null
Write-Success "uploads\, outputs\, outputs\transcode\ ready"

# =============================================================================
# DONE
# =============================================================================
Write-Host ""
Write-Host "=======================================================" -ForegroundColor White
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start the server (in this same PowerShell window):"
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python main.py"
Write-Host ""
if ($GpuMode -eq "cuda") {
    Write-Host "  GPU mode: CUDA ($CudaVer, wheels $CudaTag)"
} else {
    Write-Host "  GPU mode: CPU (slow - 5-10x real time)"
}
Write-Host ""
Write-Host "  On first run Windows Firewall will pop up asking to allow python.exe"
Write-Host "  to bind 0.0.0.0:8000. Pick 'Private networks' (Allow access)."
Write-Host ""
Write-Host "  Open  http://localhost:8000  in your browser."
Write-Host "=======================================================" -ForegroundColor White
Write-Host ""