<#
    Configure a PowerShell session to build, run and flash this project.

    Dot-source it, do not just run it - it sets environment variables that must
    survive into your shell:

        cd C:\src\tsr-esp32s3
        . .\setup_environment.ps1

    Defaults target the office machine:
        west workspace   C:\_DevTools\zephyr
        Zephyr source    C:\_DevTools\zephyr\zephyr
        project          wherever this script lives

    Overrides:
        . .\setup_environment.ps1 -ZephyrWorkspace D:\Tools\zephyrproject-v44
        . .\setup_environment.ps1 -ZephyrSdk C:\_DevTools\zephyr-sdk-1.0.1
        . .\setup_environment.ps1 -InstallPython        # create/populate a venv

    If Windows keeps asking whether to trust the file:
        Unblock-File .\setup_environment.ps1

    This script sets environment variables and PATH for the current shell only.
    Nothing is written outside the project folder unless you pass -InstallPython.
#>
[CmdletBinding()]
param(
    [string]$ZephyrWorkspace = 'C:\_DevTools\zephyr',
    [string]$ZephyrSdk       = '',
    [string]$ToolchainRoot   = '',
    [string]$Venv            = '',
    [switch]$InstallPython
)

$ErrorActionPreference = 'Stop'

# Join-Path throws on an empty root, and several of the parameters above are
# legitimately empty. Everything below goes through this instead.
function Join-IfSet {
    param([string]$Root, [string]$Leaf)
    if ([string]::IsNullOrWhiteSpace($Root)) { return $null }
    return (Join-Path $Root $Leaf)
}

$repo = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($repo)) {
    $repo = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# ---------------------------------------------------------------- Zephyr ---
if ([string]::IsNullOrWhiteSpace($ZephyrWorkspace)) {
    throw "-ZephyrWorkspace is empty. Pass the west topdir, e.g. C:\_DevTools\zephyr"
}
$ZephyrBase = Join-Path $ZephyrWorkspace 'zephyr'
if (-not (Test-Path (Join-Path $ZephyrBase 'Kconfig.zephyr'))) {
    throw ("Zephyr source not found at $ZephyrBase`n" +
           "Pass the west workspace explicitly:`n" +
           "  . .\setup_environment.ps1 -ZephyrWorkspace C:\_DevTools\zephyr`n" +
           "The workspace is the folder containing zephyr\, modules\ and .west\")
}

# ------------------------------------------------------------------- SDK ---
if ([string]::IsNullOrWhiteSpace($ZephyrSdk)) {
    $roots = @('C:\_DevTools', $HOME, 'C:\') |
             Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path $_) }

    $found = @()
    foreach ($r in $roots) {
        $found += Get-ChildItem -Path $r -Filter 'zephyr-sdk-*' -Directory `
                                -ErrorAction SilentlyContinue |
                  ForEach-Object { $_.FullName }
    }

    $candidates = @($env:ZEPHYR_SDK_INSTALL_DIR,
                    (Join-IfSet $ZephyrWorkspace 'zephyr-sdk')) + $found

    foreach ($c in $candidates) {
        if ([string]::IsNullOrWhiteSpace($c)) { continue }
        if (Test-Path (Join-Path $c 'cmake\Zephyr-sdkConfig.cmake')) {
            $ZephyrSdk = $c
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($ZephyrSdk)) {
    throw ("Zephyr SDK not found. Looked for a folder containing`n" +
           "  cmake\Zephyr-sdkConfig.cmake`n" +
           "under C:\_DevTools, your home directory and C:\.`n" +
           "Pass it explicitly:`n" +
           "  . .\setup_environment.ps1 -ZephyrSdk C:\path\to\zephyr-sdk-x.y.z")
}

# ------------------------------------------------------------------ venv ---
if ([string]::IsNullOrWhiteSpace($Venv)) {
    $wsVenv = Join-Path $ZephyrBase '.venv'
    if (Test-Path (Join-Path $wsVenv 'Scripts\python.exe')) {
        $Venv = $wsVenv
    } else {
        $Venv = Join-Path $repo '.venv'
    }
}
$python = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    if (-not $InstallPython) {
        throw ("Python venv not found at $Venv`n" +
               "Re-run with -InstallPython to create one, or point at an" +
               " existing venv with -Venv <path>.")
    }
    py -3.11 -m venv $Venv
    $python = Join-Path $Venv 'Scripts\python.exe'
}
if ($InstallPython) {
    & $python -m pip install --upgrade pip setuptools wheel
    # ai-edge-litert is the TFLite runtime verify_all.py compares the firmware
    # against - about 20 MB. Full tensorflow-cpu is ~600 MB and is only needed
    # to retrain the model or re-run the quantisation.
    & $python -m pip install numpy pillow pyserial west esptool ai-edge-litert
}

# ------------------------------------------------------------------ PATH ---
$paths = @(
    (Join-IfSet $ToolchainRoot 'cmake-3.27.6-windows-x86_64\bin'),
    (Join-IfSet $ToolchainRoot 'ninja'),
    (Join-IfSet $ZephyrSdk 'gnu\x86_64-zephyr-elf\bin'),
    (Join-IfSet $ZephyrSdk 'gnu\xtensa-espressif_esp32s3_zephyr-elf\bin'),
    (Join-IfSet $ZephyrSdk 'hosttools\qemu'),
    (Join-IfSet $ZephyrSdk 'hosttools\openocd'),
    (Join-IfSet $ZephyrBase 'scripts'),
    (Join-IfSet $Venv 'Scripts')
) | Where-Object { $_ -and (Test-Path $_) }

$env:Path = (($paths + ($env:Path -split ';')) |
             Where-Object { $_ } | Select-Object -Unique) -join ';'

$env:ZEPHYR_BASE              = $ZephyrBase
$env:ZEPHYR_SDK_INSTALL_DIR   = $ZephyrSdk
$env:ZEPHYR_TOOLCHAIN_VARIANT = 'zephyr'
# Required by the ESP32-S3 board files. Without it, cmake stops with
# "Assertion failed: ESP_IDF_PATH is not set".
$env:ESP_IDF_PATH             = Join-Path $ZephyrWorkspace 'modules\hal\espressif'

$qemu = Join-Path $ZephyrSdk 'hosttools\qemu\qemu-system-i386.exe'
if (Test-Path $qemu) { $env:QEMU_BIN = $qemu }

# ---------------------------------------------------------------- report ---
$espNote = ''
if (-not (Test-Path $env:ESP_IDF_PATH)) { $espNote = '   <-- MISSING, ESP32-S3 build will fail' }
$qemuNote = 'not found in the SDK - put qemu-system-i386 on PATH'
if ($env:QEMU_BIN) { $qemuNote = $env:QEMU_BIN }

Write-Host ''
Write-Host "project        $repo"
Write-Host "ZEPHYR_BASE    $env:ZEPHYR_BASE"
Write-Host "SDK            $env:ZEPHYR_SDK_INSTALL_DIR"
Write-Host "ESP_IDF_PATH   $env:ESP_IDF_PATH$espNote"
Write-Host "venv           $Venv"
Write-Host "QEMU           $qemuNote"
Write-Host ''

& $python --version
foreach ($t in @('cmake', 'ninja')) {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    if ($c) {
        $v = (& $t --version | Select-Object -First 1)
        Write-Host "$t           $v"
    } else {
        Write-Host "$t           NOT FOUND on PATH"
    }
}
& $python -c "import numpy, PIL, serial; print('numpy / pillow / pyserial: OK')"

# Redirecting a native command's stderr while $ErrorActionPreference is 'Stop'
# turns any stderr output into a terminating NativeCommandError on Windows
# PowerShell 5.1, so relax it just for this probe.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -c "import ai_edge_litert" 2>&1 | Out-Null
$litertOk = ($LASTEXITCODE -eq 0)
& $python -c "import tensorflow" 2>&1 | Out-Null
$tfOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEAP
if ($tfOk) {
    Write-Host 'TFLite runtime: tensorflow (verify_all.py will use this)'
} elseif ($litertOk) {
    Write-Host 'TFLite runtime: ai-edge-litert (fine - verify_all.py will use this)'
} else {
    Write-Host 'TFLite runtime: MISSING. verify_all.py cannot check bit-exactness.'
    Write-Host '                pip install ai-edge-litert     (~20 MB)'
}

Write-Host ''
Write-Host 'Next:'
Write-Host '  QEMU    cmake -B build -GNinja -DBOARD=qemu_x86 src\qemu_app ; ninja -C build'
Write-Host '  ESP32   see FLASH_ESP32S3.md'
