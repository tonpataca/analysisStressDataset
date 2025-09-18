Run MATLAB Code in Jupyter on Windows

This guide explains how to run MATLAB code inside Jupyter Lab or Notebook on Windows.

0) Prerequisites

MATLAB installed (R2018b or newer recommended).

Python 3 installed (the py launcher is fine).

JupyterLab or Jupyter Notebook installed:

py -3 -m pip install jupyterlab

1) Install the MATLAB Engine for Python

This allows Python/Jupyter to communicate with MATLAB.

Open Command Prompt as Administrator
(Start Menu → type cmd → right-click → Run as administrator)

Run the following commands (adjust the MATLAB version and path if different):

cd "C:\Program Files\MATLAB\R2025a\extern\engines\python"
py -3 -m pip install .


💡 If you're using MATLAB R2023b, for example:

cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"
py -3 -m pip install .

2) Install the MATLAB Kernel for Jupyter
py -3 -m pip install matlab_kernel

3) Launch Jupyter
jupyter lab

4) Run Your MATLAB Script

Save your MATLAB script to a known location. Example:

C:\Users\msi-pc\Desktop\Stress\dataset\extract_features.m


In a Jupyter notebook, open a MATLAB notebook (via the kernel menu) and run your .m script there.

✅ Done

You can now run MATLAB code directly in Jupyter notebooks on Windows!
