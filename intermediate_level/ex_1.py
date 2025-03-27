#ex 1
import numpy as np

a = np.random.randint(1,10,(10,10))
b = np.random.randint(1,10,(10,10))
print(a+b ,"\n", a-b, "\n ----------------------------------")

#ex 2
print(a @ b, "\n ----------------------------------")

#ex 3
print(a,"\n", np.transpose(a), "\n ----------------------------------")

#ex 4
print(np.linalg.det(a), "\n ----------------------------------")

#ex 5
c = np.random.randint(1,100,(5,5))

print(np.linalg.inv(c), "\n ----------------------------------")

#ex 6
d = np.random.randint(1,10,(10,1))
print(np.linalg.solve(a,d), "\n ----------------------------------")

#ex 7
vals,vectors = np.linalg.eig(np.random.randint(1,10,(5,5)))
print("eigenvalues:", vals, "eigenvectors:", vectors, "\n ----------------------------------")

#ex 8
print(np.linalg.matrix_rank(d), "\n ----------------------------------")

#ex 9
v1 = np.random.randint(1,10,(10,1))
v2 = np.random.randint(1,10,(10,1))
p = float(input("definire ordine norma p:\n"))
print("L_1",np.linalg.norm(v1-v2,ord=1),"\n","L_2",np.linalg.norm(v1-v2,ord=2),"\n","L_inf",np.linalg.norm(v1-v2,ord=np.inf),"\n","L_p",np.linalg.norm(v1-v2,ord=p),"\n", "\n ----------------------------------")

#ex 10
import matplotlib.pyplot as plt

x=np.linspace(-1,1,300)
plt.plot(x,np.sqrt(1-x**2),label="L2 norm", color="blue")
plt.plot(x,-np.sqrt(1-x**2), color="blue")
plt.plot(x,(1-x**4)**0.25,label="Lp norm, p=4", color="red")
plt.plot(x,-(1-x**4)**0.25, color="red")
plt.plot(x,(1-np.abs(x)),label="L1 norm", color="green")
plt.plot(x,-(1-np.abs(x)), color="green")
plt.plot([1, -1, -1, 1, 1],[1, 1, -1, -1, 1],label="L_inf norm", color="grey")
plt.legend(loc='lower right')
plt.axis('equal')
plt.show()

#ex 11
import math
v1=np.random.rand(5)
v2=np.random.rand(5)
print(math.acos(np.dot(v1,v2)/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2,ord=2))), "\n ----------------------------------")

#ex 12
print(np.linalg.norm(a, 'fro'),"\n----------------------------------")