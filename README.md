Welcome to PyDial: the CUED Python Statistical Dialog System (V1.4)
If you are reading this you have probably already downloaded the PyDial repo but if
you havent, do it now.

PyDial uses Python 3.7 and has been previously implemented with python 2.7. Ensure
that you use the requirements.txt file to install the appropriate dependencies
via pip. If you do not have pip installed yet first do

sudo apt-get install python3-pip

otherwise exectued directly

pip3 install -r requirements.txt

To check that you have a fully functioning system, run the PyDial functional tests

sh testPyDial

You should see output similar to the following:
Running PyDial Tests
1 tests/test_DialogueServer.py   time 0m3.908s
2 tests/test_Simulate.py         time 0m18.990s
3 tests/test_Tasks.py            time 0m0.492s
3 tests: 980 warnings,   0 errors
See test logs in _testlogs for details
Finally, install the documentation

sh createDocs.sh

Then point your browser at documentation/Docs/index.html.  If PyDial is new to you,
read the Tutorial "Introduction to PyDial".
The PyDial Team
August 2018
PyDial is distributed under Apache 2.0 Licensed. See LICENSE