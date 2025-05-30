@echo off
cd /d "%~dp0\.."
"C:\Program Files\Git\bin\bash.exe" -c "source ~/.bashrc && conda activate segmentation && python -m microseg.image.segment %*" 