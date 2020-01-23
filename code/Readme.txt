This is the instruction on how to run 3D TV with convolutional saprse representation.
1. Install sporco following instruction here: https://sporco.readthedocs.io/en/latest/install.html
2. Open test-cpbdn.py, change line 50 to: mu = 0.017(optimized parameter for 2D TV), run the code to see 2D TV results.
3. repalce the file in sporco\linalg.py and sporco\admm\cpbdntv.py by the corresponding one inside this folder with exaclty same name.
4. Open test-cpbdn.py, change line 50 to: mu = 0.005(optimized parameter for 3D TV), run the code to see 3D TV results.

If you have any question please contact: f.long@wustl.edu. I'll see if I could help with that.