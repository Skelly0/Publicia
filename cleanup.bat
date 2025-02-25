@echo off
echo Cleaning up the repository and removing sensitive files...

REM Remove files from Git's tracking but keep them locally
git rm -r --cached .
git add .
git commit -m "Apply gitignore rules and remove sensitive data"
git push

echo.
echo Repository cleaned! Sensitive files are no longer tracked but remain on your local machine.
echo.
pause
