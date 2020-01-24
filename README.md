# Convolutional-Sparse-Representation-with-3D-Total-Variation

## overview

 Studied an efficient way to solve convolutional Sparse coding problem.
 Solve convolutional sparse coding with different types of 2D TV,
 extend it to implement 3D TV, test them together with traditional convolutional sparse coding and patch-wised sparse cod- ing on the situation of Gaussian white noise restoration situation. All the tests and code is based on sporco[5] library.




## Installing&&running

1. Install sporco following instruction here: https://sporco.readthedocs.io/en/latest/install.html
2. Open test-cpbdn.py, change line 50 to: mu = 0.017(optimized parameter for 2D TV), run the code to see 2D TV results.
3. repalce the file in sporco\linalg.py and sporco\admm\cpbdntv.py by the corresponding one inside this folder with exaclty same name.
4. Open test-cpbdn.py, change line 50 to: mu = 0.005(optimized parameter for 3D TV), run the code to see 3D TV results.



## result:
Check out final report in report repo for more details.
![image](https://github.com/Robert19950512/Convolutional-Sparse-Representation-with-3D-Total-Variation/blob/master/web/test-result1.png)



author: Fa Long
       2020.1
