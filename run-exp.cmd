:::::
:: Run a batch experiment
:: ver: 20210923.2001
:::::
:: Do not display every line of the code
@echo off

:: Make sure all variables in this script is local
setlocal

:: ----- Config : begin

:: Directory for input files
set "EXP_INPUT_DIR=input"

:: Output file for recording results
set "EXP_OUTPUT=output\calcInfo.csv"

:: Command to call for each input file
set "EXP_CMD=python calcInfo.py"
:: ----- Config : end

:: Get this script's directory
for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"

:: Go into the script's directory (incase this script is called from other directory)
pushd %SCRIPT_DIR%

:: Run EXP_CMD on each file in EXP_INPUT_DIR
for %%f in ("%EXP_INPUT_DIR%\*.*") do (
    echo Processing "%EXP_INPUT_DIR%\%%~nxf" ...
    call %EXP_CMD% "%EXP_INPUT_DIR%\%%~nxf" "%EXP_OUTPUT%"
)

:: Return to the previous directory
popd

:: Exit
endlocal & exit /b
