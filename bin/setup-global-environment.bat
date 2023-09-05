@echo off
setlocal
set MAMBA_NO_BANNER=1
set PROMPT_PREFIX=[33m
set SUCCESS_PROMPT_PREFIX=[92m
set ERROR_PROMPT_PREFIX=[91m
set PROMPT_SUFFIX=[0m

echo %PROMPT_PREFIX%Setting environment variables...%PROMPT_SUFFIX%
setx MAMBA_NO_BANNER 1 || goto :error

echo %PROMPT_PREFIX%Setting PowerShell execution policy...%PROMPT_SUFFIX%
powershell -NoProfile -Command "Set-ExecutionPolicy -Force -ExecutionPolicy RemoteSigned -Scope CurrentUser" || goto :error

echo %PROMPT_PREFIX%Making sure that Winget is available...%PROMPT_SUFFIX%
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.DesktopAppInstaller_8wekyb3d8bbwe" || goto :error

echo %PROMPT_PREFIX%Installing Mambaforge if needed...%PROMPT_SUFFIX%
winget list mambaforge || winget install mambaforge -s winget || goto :error

echo %PROMPT_PREFIX%Setting up Conda base environment...%PROMPT_SUFFIX%
REM Conda updates often break stuff :<
call %LOCALAPPDATA%\mambaforge\condabin\conda.bat config --set auto_update_conda false || goto :error
call %LOCALAPPDATA%\mambaforge\condabin\mamba.bat install -n base -y conda-lock conda-libmamba-solver invoke || goto :error
REM Use libmamba solver for *much* faster dependency resolution
call %LOCALAPPDATA%\mambaforge\condabin\conda.bat config --set solver libmamba || goto :error

echo %PROMPT_PREFIX%Initializing Conda for all shells...%PROMPT_SUFFIX%
call %LOCALAPPDATA%\mambaforge\condabin\conda.bat init || goto :error

echo %SUCCESS_PROMPT_PREFIX%Done! Open a new PowerShell/cmd/bash prompt to start using Mambaforge.%PROMPT_SUFFIX%
pause
goto :EOF

:error
echo %ERROR_PROMPT_PREFIX%Oh noes! Something went wrong.%PROMPT_SUFFIX%
pause
