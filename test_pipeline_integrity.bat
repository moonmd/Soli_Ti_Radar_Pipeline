@echo off
setlocal

echo mmWave to Soli-format Converter Test Script
echo ===========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.10 and try again.
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
pip show numpy scipy h5py matplotlib >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Missing required packages.
    echo Please install required packages:
    echo pip install -r requirements.txt
    exit /b 1
)

echo.
echo Clearing Frame Dumps...
del /Q dumps\*frame*.npy

echo.
echo Step 1: Generating synthetic test data...
python generate_test_data.py --cfg example_config.txt --out test_data.bin --frames 100 --gesture --no-noise --debug
::python generate_test_data.py --cfg example_config.txt --out test_data.bin --frames 100 --gesture %SHOW%
if %errorlevel% neq 0 (
    echo Error: Failed to generate test data.
    exit /b 1
)

echo.
echo Step 2: Converting test data to Soli format...
python mmwave_to_soli.py --cfg example_config.txt --bin test_data.bin --out test_gesture.h5 --label 1 --frames 40 --debug
if %errorlevel% neq 0 (
    echo Error: Failed to convert test data.
    exit /b 1
)

echo.
echo Step 3: Verify synthetic ADC exported/imported frame alignment
python verify_export.py
if %errorlevel% neq 0 (
    echo Error: ADC export/import verification failed.
    exit /b 1
)

echo.
echo Step 4: Verifying the HDF5 output file...
python verify_hdf5.py --file test_gesture.h5
if %errorlevel% neq 0 (
    echo Error: HDF5 verification failed.
    exit /b 1
)

echo.
echo Step 5: Verifying RDI dumps...
python verify_RDI.py
if %errorlevel% neq 0 (
    echo Error: RDI verification failed.
    exit /b 1
)

echo.
echo Step 6: Generating RDI GIF animation...
python .\rdgifh5.py .\test_gesture.h5
if %errorlevel% neq 0 (
    echo Error: RDI GIF generation failed.
    exit /b 1
)

echo.
echo Test completed successfully!
echo Output file: test_gesture.h5
echo.
endlocal
