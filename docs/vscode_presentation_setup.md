# VS Code setup for the college demonstration

Prepared 26 October 2026. VS Code is the editor; Python, dependencies, trained weights and CARLA are separate runtime requirements.

## If this is the same Lenovo laptop already prepared

1. Download the Windows User Installer from https://code.visualstudio.com/docs/setup/windows and install VS Code.
2. Open Extensions with Ctrl+Shift+X. Install **Python**, published by Microsoft (ms-python.python). **Pylance** by Microsoft (ms-python.vscode-pylance) is optional for code completion and may be offered with Python. No extra extension is required to launch the demo from the terminal.
3. Use File > Open Folder and open the existing FinalProject folder:
   C:\Users\Lenovo\Documents\Codex\2026-10-24\https-github-com-pjha26-finalproject-https\FinalProject
4. Press Ctrl+Shift+P, choose Python: Select Interpreter, then Enter interpreter path. Select the existing .venv\Scripts\python.exe inside that folder. Keep the prepared Python 3.8.20 environment.
5. Open Terminal > New Terminal (PowerShell). Run from the project root:

   .\tools\present.ps1 -Mode carla -CheckOnly
   .\tools\present.ps1 -Mode carla

6. Rehearse the fallback separately:

   .\tools\present.ps1 -Mode video

The launcher uses the explicit local interpreter. Environment activation and the editor's Run/Debug button are unnecessary for this demonstration. Current debugger extensions can have separate Python-version requirements; use the terminal commands for this pinned Python 3.8 project.

If PowerShell blocks the script, use:

    $env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
    .\.venv\Scripts\python.exe tools\present.py --mode carla

Replace carla with video for the recorded-input fallback. Each run gets a fresh output folder automatically. Plug in power, close other GPU-heavy programs and let the full 65-second CARLA rehearsal finish. Close any existing CARLA server before using the launcher.

## If this is a different friend's laptop

Installing VS Code and its Python extension alone will not prepare it. Clone the repository or transfer the source, then rebuild a compatible Python 3.8 environment and install requirements.txt. Python packages such as PyTorch and OpenCV are pip dependencies, not VS Code extensions. Install a compatible NVIDIA driver for the CUDA demonstration.

Transfer or prepare the accepted checkpoint folder, pinned third_party/monodepth2 source and licenses, data/demo assets and the CARLA 0.9.13 server at .runtime/carla/. Keep folder layouts and manifests intact. Rebuild .venv on the new computer; copying a virtual environment is not a reliable handoff. A full delivery check also needs data/fixtures and data/evaluation/redwood. Follow docs/demo_guide.md and project.py's preparation commands. Do not assume the twelve-FPS result transfers to different hardware.

Before using that laptop, run:

    .\.venv\Scripts\python.exe tools\validate_delivery.py --device cuda --display

Then run both presentation commands above. Check actual saved summaries. If setup cannot be completed and rehearsed tonight, use the already prepared Lenovo laptop.

## Official setup references

- Windows installation: https://code.visualstudio.com/docs/setup/windows
- Microsoft Python extension: https://marketplace.visualstudio.com/items?itemName=ms-python.python
- Pylance: https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance
- Interpreter selection: https://code.visualstudio.com/docs/python/environments
- Debugger/version information: https://code.visualstudio.com/docs/python/debugging

