param([ValidateSet('carla','video')][string]$Mode = 'carla', [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache/matplotlib'
$arguments = @((Join-Path $PSScriptRoot 'present.py'), '--mode', $Mode)
if ($CheckOnly) { $arguments += '--check-only' }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') @arguments
if ($LASTEXITCODE -ne 0) { throw 'Presentation failed. Inspect the retained run/server log and docs/presentation_runbook.md.' }


