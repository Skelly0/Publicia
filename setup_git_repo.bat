@echo off
echo Setting up Git repository for Publicia Bot...

REM Copy main files to the repository
echo Copying files to repository...
xcopy /Y "C:\Users\samwi\OneDrive\Publicia\PubliciaV5.py" "C:\Users\samwi\Publicia-Bot\"
xcopy /Y "C:\Users\samwi\OneDrive\Publicia\system_prompt.py" "C:\Users\samwi\Publicia-Bot\"
xcopy /Y "C:\Users\samwi\OneDrive\Publicia\.gitignore" "C:\Users\samwi\Publicia-Bot\"

REM Create a requirements.txt file
echo Creating requirements.txt file...
echo discord.py>=2.0.0 > "C:\Users\samwi\Publicia-Bot\requirements.txt"
echo python-dotenv >> "C:\Users\samwi\Publicia-Bot\requirements.txt"
echo sentence-transformers >> "C:\Users\samwi\Publicia-Bot\requirements.txt"
echo numpy >> "C:\Users\samwi\Publicia-Bot\requirements.txt"
echo aiohttp >> "C:\Users\samwi\Publicia-Bot\requirements.txt"
echo apscheduler >> "C:\Users\samwi\Publicia-Bot\requirements.txt"

REM Create a documentation directory
echo Creating documentation...
mkdir "C:\Users\samwi\Publicia-Bot\docs"
xcopy /Y "C:\Users\samwi\OneDrive\Publicia\publicia-documentation.md" "C:\Users\samwi\Publicia-Bot\docs\"

REM Move to the repository directory
cd "C:\Users\samwi\Publicia-Bot"

REM Add files to Git
echo Adding files to Git...
git add .

REM Commit the files
echo Committing files...
git commit -m "Add Publicia bot initial files"

REM Push to GitHub
echo Pushing to GitHub...
git push origin main

echo Done! Repository has been set up and pushed to GitHub.
