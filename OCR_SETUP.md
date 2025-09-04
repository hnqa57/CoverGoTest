# Quick OCR Setup Guide

## Install Tesseract OCR

### Windows (Recommended)

1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. During installation, check "Additional language data" for Chinese support
3. Add to PATH or note installation directory

### PowerShell Commands

```powershell
# Option 1: Using winget (Windows 10/11)
winget install --id UB-Mannheim.Tesseract -e

# Option 2: Using Chocolatey
choco install tesseract -y

# Verify installation
tesseract --version
```

## Test OCR

```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

## For Chinese/Multilingual Documents

The app automatically detects and uses appropriate languages:

- English: `eng`
- Chinese Simplified: `chi_sim`
- Chinese Traditional: `chi_tra`
- Combined: `eng+chi_sim+chi_tra`

## Manual Tesseract Path (if needed)

If Tesseract isn't in PATH, specify the full path:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```
