@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%pure_c_kedge_cuda.exe"
set "OPENJPEG_INCLUDE=D:\python\pkgs\openjpeg-2.5.2-h9b5d1b5_1\Library\include\openjpeg-2.5"
set "OPENJPEG_LIB=D:\python\Library\lib\openjp2.lib"
set "OPENJPEG_DLL=D:\python\Library\bin\openjp2.dll"
set "NVJPEG2K_ROOT=C:\Program Files\NVIDIA nvJPEG2K\v0.10"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
set "VSVARS="
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%NVCC%" (
    echo nvcc not found: %NVCC%
    exit /b 1
)
if not exist "%OPENJPEG_INCLUDE%\openjpeg.h" (
    echo openjpeg.h not found: %OPENJPEG_INCLUDE%
    exit /b 1
)
if not exist "%OPENJPEG_LIB%" (
    echo openjp2.lib not found: %OPENJPEG_LIB%
    exit /b 1
)
if not exist "%OPENJPEG_DLL%" (
    echo openjp2.dll not found: %OPENJPEG_DLL%
    exit /b 1
)
if not exist "%NVJPEG2K_ROOT%\include\nvjpeg2k.h" (
    echo nvjpeg2k.h not found: %NVJPEG2K_ROOT%\include\nvjpeg2k.h
    exit /b 1
)

if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
            set "VSVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)

if not defined VSVARS (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set "VSVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if not defined VSVARS (
    echo vcvars64.bat not found. Please install Visual Studio C++ tools first.
    exit /b 1
)

call "%VSVARS%"
if not %ERRORLEVEL%==0 (
    echo Failed to initialize MSVC environment.
    exit /b 1
)

"%NVCC%" ^
  -I "%OPENJPEG_INCLUDE%" ^
  -I "%NVJPEG2K_ROOT%\include" ^
  -O3 ^
  --use_fast_math ^
  -arch=sm_86 ^
  -Xptxas=-O3,-dlcm=ca ^
  -std=c++17 ^
  -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT ^
  -DCUB_IGNORE_DEPRECATED_CPP_DIALECT ^
  -Xcompiler="/utf-8 /W3 /EHsc /O2 /Ot /fp:fast /D_CRT_SECURE_NO_WARNINGS" ^
  -DKEDGE_USE_CUDA ^
  "%ROOT%main.c" ^
  "%ROOT%dicom_io.c" ^
  "%ROOT%jp2_decode.c" ^
  "%ROOT%decomposition.c" ^
  "%ROOT%dicom_export.c" ^
  "%ROOT%kedge_common.c" ^
  "%ROOT%cuda_backend.cu" ^
  "%ROOT%cuda_kernels.cu" ^
  "%ROOT%nvjp2_decode.cu" ^
  "%OPENJPEG_LIB%" ^
  "%NVJPEG2K_ROOT%\lib\12\nvjpeg2k.lib" ^
  -o "%OUT%"

if %ERRORLEVEL%==0 (
    copy /Y "%OPENJPEG_DLL%" "%ROOT%openjp2.dll" >nul
    copy /Y "%NVJPEG2K_ROOT%\bin\12\nvjpeg2k_0.dll" "%ROOT%nvjpeg2k_0.dll" >nul
    echo Built: %OUT%
)

exit /b %ERRORLEVEL%
