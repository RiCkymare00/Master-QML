import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Reals, maximize, Binary, minimize
ipopt_path = "/opt/homebrew/bin/ipopt"

'''
/           task1   task2   task3   task4
machine1    13      4       7       6
machine2    1       11      5       4    
machine3    6       7       2       8
machine4    1       3       5       9
'''
#minimize 13*model.x[1,1]+4*model.x[1,2]+7*model.x[1,3]+6*model.x[1,4]+model.x[2,1]+11*model.x[2,2]+5*model.x[2,3]+4*model.x[2,4]+6*model.x[3,1]+7*model.x[3,2]+2*model.x[3,3]+8*model.x[3,4]+model.x[4,1]+3*model.x[4,2]+5*model.x[4,3]+9*model.x[4,4]
#         sum((model.x[i,j] for j in [1,2,3,4]) for i in [1,2,3,4]) == 1  
#         sum((model.x[j,i] for j in [1,2,3,4]) for i in [1,2,3,4]) == 1
#         ((model.x[i,j] for j in [1,2,3,4]) for i in [1,2,3,4]) >= 0    

model = ConcreteModel()
model.x = Var([1,2,3,4],[1,2,3,4], domain = Binary, initialize=lambda model, i, j: np.random.randint(2))
print('Valori inziali \n')
print("\n".join(f"x[{i},{j}] = {model.x[i,j].value}" for i in [1,2,3,4] for j in [1,2,3,4]))
model.OBJ = Objective(expr = 13*model.x[1,1]+4*model.x[1,2]+7*model.x[1,3]+6*model.x[1,4]+model.x[2,1]+11*model.x[2,2]+5*model.x[2,3]+4*model.x[2,4]+6*model.x[3,1]+7*model.x[3,2]+2*model.x[3,3]+8*model.x[3,4]+model.x[4,1]+3*model.x[4,2]+5*model.x[4,3]+9*model.x[4,4], sense = minimize)
model.Constraint1 = Constraint([1,2,3,4], rule=lambda model, i: sum(model.x[i, j] for j in [1,2,3,4]) == 1)
model.Constraint2 = Constraint([1,2,3,4], rule=lambda model, i: sum(model.x[j, i] for j in [1,2,3,4]) == 1)
model.Constraint3 = Constraint([(i, j) for i in [1,2,3,4] for j in [1,2,3,4]],rule=lambda model, i, j: model.x[i, j] >= 0)
solver = SolverFactory('ipopt', executable = ipopt_path)
solver.options['max_iter'] = 1000
solver.solve(model,tee = False)
print('Valori ottimizzati')
#print('\n'.join(f'x[{i,j}] = {model.x[i,j].value}'for i in [1,2,3,4] for j in [1,2,3,4]))
print("\n".join(f"x[{i},{j}] = {model.x[i,j].value}" for i in [1, 2, 3, 4] for j in [1, 2, 3, 4] if model.x[i, j].value > 0.9)) 

