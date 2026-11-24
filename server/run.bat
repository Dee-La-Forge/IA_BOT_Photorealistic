@echo off
rem Lance le serveur local (avatar, ecoute, voix). Rien n'est envoye sur Internet.
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
if not defined VISAVIS_ROOT set VISAVIS_ROOT=D:\visavis
set FORCING_DIR=%VISAVIS_ROOT%\AvatarForcing
set VISAVIS_DATA=%VISAVIS_ROOT%\data
rem Fichiers temporaires sur le meme disque que les modeles, pas sur C:.
set TMP=%VISAVIS_ROOT%\cache\tmp
set TEMP=%VISAVIS_ROOT%\cache\tmp
cd /d "%~dp0"
"%VISAVIS_ROOT%\env\python.exe" main.py
