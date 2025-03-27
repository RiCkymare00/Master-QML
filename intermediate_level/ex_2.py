import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Reals, maximize
ipopt_path = "/opt/homebrew/bin/ipopt"

#maximize -x**2-3*(x**2)*y-2*y**2-z**3
#subject to x + y**2 >= 2
#           3*y <= 3
#           z**2 >= 4

x=[]
y=[]
z=[]
val=[]
k=0
while k < 50:
    print('massimizzazione numero', k, '\n')
 
    valid = False
    while not valid:
        init_x1 = np.random.randint(4)
        init_x2 = np.random.randint(4)
        init_x3 = np.random.randint(4)
        if (init_x1 + init_x2 >= 2) and (3*init_x2 <= 3) and (init_x3**2 >= 4):
            valid = True
        else:
            print('Punto iniziale non valido, genero un nuovo punto...')
    
        model = ConcreteModel()
        model.x = Var([1, 2, 3], domain=Reals, initialize={1: init_x1, 2: init_x2, 3: init_x3})
        
        print("Initial values:")
        for l in [1, 2, 3]:
            print(f"x[{l}] = {model.x[l].value}")
        
        model.OBJ = Objective(expr = -model.x[1]**2 - 3*(model.x[1]**2)*model.x[2] - 2*model.x[2]**2 - model.x[3]**3, 
                            sense= maximize)
        model.Constraint1 = Constraint(expr = model.x[1] + model.x[2] >= 2)
        model.Constraint2 = Constraint(expr = 3*model.x[2] <= 3)
        model.Constraint3 = Constraint(expr = model.x[3]**2 >= 4)

        solver = SolverFactory('ipopt', executable=ipopt_path)
        solver.options['max_iter'] = 1000
        try:
            solver.solve(model, tee=False)
        except:
            print('Generazione di un nuovo punto iniziale...')
            continue

        print("Optimal values:")
        for i in [1, 2, 3]:
            print(f"x[{i}] = {model.x[i].value}")
        
        x.append(model.x[1].value)
        y.append(model.x[2].value)
        z.append(model.x[3].value)
        val.append(-model.x[1].value**2 - 3*(model.x[1].value**2)*model.x[2].value - 2*model.x[2].value**2 - model.x[3].value**3)
        
        k += 1

if x and y and z and val:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    scatter = ax.scatter(x, y, z, c=val, cmap='RdYlGn') 
    fig.colorbar(scatter, ax=ax, label='computational value')
    plt.show()
else:
    print('Nessun punto valido')