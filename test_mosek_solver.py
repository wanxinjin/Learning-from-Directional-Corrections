import cvxpy as cp
import numpy as np
# this code is to test if your MOSEK solver has been properly installed
# the problem this code is to solve is the following semi-definite programming
# min_{C,d} -log(det(C))
# subject to: norm_2(C*a_i)+a_i'*d <b_i, i=1,2,3,4
# where a_1=[-1, 0], b_1=1
#       a_2=[ 1, 0], b_2=1
#       a_3=[ 0, 1], b_3=1
#       a_4=[ 0,-1], b_4=1


# define a symmetric matrix variable C of dimension 2x2
C = cp.Variable((2, 2), symmetric=True)
# define a vector variable d of dimension 2
d = cp.Variable(2)

# create the constraints
a=np.array([[0,1], [0,-1], [1,0], [-1,0]])   # a_i
b=np.array([1,1,1,1]) # b_i
constraints = [C >> 0]
for i in range(a.shape[0]):
    constraints += [(cp.norm2(C @ a[i].reshape(-1, 1)) + a[i] @ d) <= b[i]]

# establish the problem and solve it
prob = cp.Problem(cp.Minimize(-cp.log_det(C)),
                  constraints)
prob.solve(solver=cp.MOSEK, verbose=False)

# print your results
print(C.value, d.value)
print("Congratulations! You passed the test and you have correctly installed mosek!")

# if your reults are C.value=[1,0, 0, 1], d.value=[0,0], then great, you have correctly installed mosek
# otherwise, if you have an error when run, you have not correctly installed it. please check the error to fix it.