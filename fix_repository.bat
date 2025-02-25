@echo off
echo This script will fix your repository tracking and clean up sensitive files
echo.

REM Move the new gitignore to replace the current one
echo Updating .gitignore file...
copy .gitignore.new .gitignore

REM Remove sensitive files from Git tracking (but keep them locally)
echo Removing sensitive files from Git tracking...
git rm -r --cached .
git add .
git commit -m "Properly apply gitignore rules"

echo.
echo Repository fixed! Now push to GitHub with:
echo git push
echo.
echo After pushing, you may need to change your repository visibility on GitHub.
pause
