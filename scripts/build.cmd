@echo off
setlocal enabledelayedexpansion

set KAIN_BIN="T:\KainProjects\Kain\.kain\bin\kain.exe"
if not exist %KAIN_BIN% (
    set KAIN_BIN=kain
)

echo ================================================================================
echo  juicer.kn — Build Pipeline
echo ================================================================================

cd /d "%~dp0.."

echo [1/3] Amalgamating kain/core/*.kn -^> kain/juicer.kn ...
%KAIN_BIN% amalgamate --raw kain/core -o kain/juicer.kn
if %errorlevel% neq 0 (
    echo [ERROR] Amalgamation failed.
    exit /b %errorlevel%
)

echo [2/3] Compiling native LLVM binary juicer.exe ...
%KAIN_BIN% build kain/juicer.kn --target llvm -o juicer.exe
if %errorlevel% neq 0 (
    echo [ERROR] Native compilation failed.
    exit /b %errorlevel%
)

echo [3/3] Emitting CLI alias jc.exe ...
copy /y juicer.exe jc.exe >nul

echo ================================================================================
echo  Build SUCCESS: juicer.exe ^& jc.exe ready!
echo ================================================================================
