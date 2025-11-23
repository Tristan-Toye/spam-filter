
@echo off
setlocal

cd build || exit /b 1
cmake .. || exit /b 1
cmake --build . || exit /b 1
cd .. || exit /b 1
build\src\Debug\bdap_assignment1.exe || exit /b 1
