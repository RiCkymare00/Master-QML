import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Reals, maximize, Binary
ipopt_path = "/opt/homebrew/bin/ipopt"

'''
item        weight  rating
ant-rep     1       2
beer        3       9
blanket     4       3
bratwurst   3       8
brownies    3       10
friesbee    1       6
salad       5       4
watermelon  10      10 
'''
#maximize 2*x[1] + 9*x[2] + 3*x[3] + 8*x[4] + 10*x[5] + 6*x[6] + 4*x[7] + 10*x[8]
#         x[1] + 3*x[2] + 4*x[3] + 3*x[4] + 3*x[5] + x[6] + 5*x[7] + 10*x[8] <= 15
#         x[i] >= 0 

#This model is a binary linear programming model. It is linear because the objective function and the constraints are linear functions of the decision variables. It is binary because the decision variables are restricted to be either 0 or 1.

for l in range(200):
    model = ConcreteModel()
    model.x = Var([1,2,3,4,5,6,7,8], domain=Binary, initialize={i: np.random.randint(2) for i in [1,2,3,4,5,6,7,8]})
    '''print("Valori iniziali: \n")
    for j in [1,2,3,4,5,6,7,8]:
        print(model.x[j].value)'''
    model.OBJ = Objective(expr = 2*model.x[1] + 9*model.x[2] + 3*model.x[3] + 8*model.x[4] + 10*model.x[5] + 6*model.x[6] + 4*model.x[7] + 10*model.x[8], sense = maximize)
    model.Constraint1 = Constraint(expr = model.x[1] + 3*model.x[2] + 4*model.x[3] + 3*model.x[4] + 3*model.x[5] + model.x[6] + 5*model.x[7] + 10*model.x[8] <= 15)
    solver = SolverFactory('ipopt', executable=ipopt_path)
    solver.options['max_iter'] = 1000
    solver.solve(model, tee=False)
    print("Valori ottimizzati: \n")
    for j in [1,2,3,4,5,6,7,8]:
        print(f"x[{j}] = {model.x[j].value}")