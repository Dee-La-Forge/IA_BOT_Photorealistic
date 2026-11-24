@echo off
rem Installation complete du serveur local (environ 20 Go, a faire une seule fois).
rem Prerequis : conda, git, une carte NVIDIA, et Ollama avec un modele installe.
setlocal
if not defined VISAVIS_ROOT set VISAVIS_ROOT=D:\visavis
set FLASHHEAD_DIR=%VISAVIS_ROOT%\SoulX-FlashHead
set VISAVIS_DATA=%VISAVIS_ROOT%\data
set PIP_CACHE_DIR=%VISAVIS_ROOT%\cache\pip
set TMP=%VISAVIS_ROOT%\cache\tmp
set TEMP=%VISAVIS_ROOT%\cache\tmp
set PY="%VISAVIS_ROOT%\env-fh\python.exe"

if not exist "%TMP%" mkdir "%TMP%"
if not exist "%FLASHHEAD_DIR%" git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git "%FLASHHEAD_DIR%" || goto :error
if not exist "%VISAVIS_ROOT%\env-fh\python.exe" call conda create -y -p "%VISAVIS_ROOT%\env-fh" python=3.10 || goto :error

%PY% -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 || goto :error
rem Sans nvidia-nccl ni xfuser : ils ne servent qu'au calcul sur plusieurs cartes, sous Linux.
findstr /v /i "nvidia-nccl xfuser" "%FLASHHEAD_DIR%\requirements.txt" > "%TMP%\flashhead-requirements.txt"
%PY% -m pip install -r "%TMP%\flashhead-requirements.txt" || goto :error
%PY% -m pip install -r "%~dp0requirements.txt" || goto :error
%PY% "%~dp0download_models.py" || goto :error

echo.
echo Installation terminee. Lancez start.bat a la racine du projet.
exit /b 0

:error
echo.
echo L'installation a echoue a l'etape ci-dessus.
exit /b 1
