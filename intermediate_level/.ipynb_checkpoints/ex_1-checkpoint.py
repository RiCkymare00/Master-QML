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

L_1 = []
for i in range(100):
    v = np.random.rand(1, 2) * 2 - 1
    if np.linalg.norm(v, ord = 1) == 1:
        L_1.append(v)

plt.plot(np.linspace(-2,2),np.linspace(-2,2),L_1)
plt.show()