@echo off
rem Lance le serveur local dans une fenetre, puis l'interface web.
cd /d "%~dp0"
start "Vis-a-vis - serveur local" cmd /k server\run.bat
start "" http://localhost:3000
npm run dev
