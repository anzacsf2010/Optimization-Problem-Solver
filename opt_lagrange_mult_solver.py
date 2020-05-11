#!/usr/bin/python3
#-------------------------------------------#
"""
###########################
+++++++
Script:
+++++++
A solver for solving an optimization problem using Lagrange Multiplier method 
with strict user inputs for the objective function and all constraints.

+++
By: 
+++
Andy St. Fort

++++++
Logic: 
++++++
1. An objective function in the form of ax^n + by^m is given by the user. 

2. Then, the script asks the user for how many constraints (up to 5) will be used as part of the equation. 

3. The script uses that value to ask the user to enter each of the constraint equations in terms of x and y. 

4. The script solves the optimization problem using the sympy tools and output the result to the user. 
###########################
"""
print(__doc__)
# Importing the necessary packages for this to work
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sympy as sp
from sympy.plotting import plot, plot3d
from sympy import init_printing
init_printing(use_unicode=True)

# Main method to be called against later
def main():
    """
    Main method used to take raw input from user and then tranform into an equation
    using sympy along with the needed constraints, also given by the user.
    ++++++++++++++++++++++++++++++++++++
    What this script does is as follows:
    ++++++++++++++++++++++++++++++++++++
    1. Specify the Lagrangian function for the problem using the given objective and constraints equations
    2. Determine the Karush-Kuhn-Tucker (KKT) conditions
    3. Find the (x,y) tuples that satisfy the KKT conditions
    4. Determine which of these (x,y) tuples correspond to the minimum/maximum of f(x,y)
    """
    print("#################################")
    print("Problem Inputs:")
    print("#################################")
    # Setting up the variables
    x, y = sp.symbols('x y')

    # Ask user to enter the main objective function
    # The logic here is requiring that the user inputs a non-empty objective function and allow him/her to decide whether that value was correct using yes (y) or no (n)
    eq_1 = input("Enter your objective function here, in terms of x and y, in the form of ax^n + by^m + c, with a, b, c, n, and m being constants:")
    while not eq_1:
        print('Objective function cannot be empty.')
        eq_1 = input("Enter your objective function here, in terms of x and y:")
    print(f'You have entered {eq_1}.')
    obj = ''
    continue_decision_var = input("Enter y for continue or n to re-enter the objective function:")
    while (continue_decision_var != 'y'):
        if (continue_decision_var == 'n'):
            eq_1 = input("Enter your objective function here, in terms of x and y:") 
            while not eq_1:
                print('Objective function cannot be empty.')
                eq_1 = input("Enter your objective function here, in terms of x and y:")
            print(f'You have entered {eq_1}.')
            continue_decision_var = input("Enter y for continue or n to re-enter the objective function:")
        else:
            print("Value needs to be y or n.")
            continue_decision_var = input("Enter y for continue or n to re-enter the objective function:")
    obj_function = sp.sympify(eq_1,evaluate=False)
    sp.pprint(obj_function)

    # Ask the user whether the problem is to maximize or minimize the objective function
    while True:
        try:
            optimization_option = input("Enter m for minimize or M to maximize the objective function:")
        except ValueError:
            print("Choice has to be m or M.")
            continue
        if not optimization_option:
            print("Choice has to be m or M.")
            continue
        if ((optimization_option != 'm') and (optimization_option != 'M')):
            print("Choice has to be m or M.")
            continue
        if (optimization_option == 'm'):
            print("You have chosen to minimize the objective function.")
            optimization_decision = 'minimize'
            break
        if (optimization_option == 'M'):
            print("You have chosen to Maximize the objective function.")
            optimization_decision = 'maximize'
            break
        return optimization_decision

    # Ask user for how many constraints will be used in the optimization problem
    # The logic here requires the user that an integer larger than 0 be used for number of constraints. It also checks that the value is stricly an integer.
    while True:
        try:
            number_of_constraints = int(input("How many constraints will be used for this optimization problem? At least 1 constraint is required, and up to 5 can be used:"))
        except ValueError:
            print("Number of constraints must be an integer.")
            continue
        if ((number_of_constraints < 1) or (number_of_constraints > 5)):
            print("Number of constraints must be an integer between 1 and 5.")
            continue
        else:
            print(f'{number_of_constraints} constraints will be used.')
            number_of_constraints = number_of_constraints
            break 
        return number_of_constraints

    # Use the amount of constraints to get each constraint input and save as a sympy object
    # The logic here is to repeat the operation for each constraint until the number of {number_of_constraints} is reached.
    constraints_counter = 0
    constraint = dict()
    while (constraints_counter < number_of_constraints):
        while True:
            try:
                constraint_eq = input(f'Enter constraint {constraints_counter + 1} here, in the form of ax^n + by^m + c, with a, b, c, n, and m being constants:')
            except ValueError:
                print("Invalid entry for the constraint equation.")
                continue
            if constraint_eq == '':
                print("Constraint equation cannot be empty.")
                continue
            else:
                constraint_eq = sp.sympify(constraint_eq,evaluate=False)
                print(f'Constraint {constraints_counter + 1} is:')
                constraint[constraints_counter] = constraint_eq
                sp.pprint(sp.Eq(constraint[constraints_counter]))
                break
            return constraint[constraints_counter]
        constraints_counter = constraints_counter + 1

    # Solve the problem
    # Here, we use the sympy.solve function to solve the problem, but we also ensure that all (up to 5) constraints are accounted for. We consider 5 if statements, each representing a choice from the user about the number of constraints being used.
    print("#################################")
    print("Problem Solution:")
    print("#################################")
    def solve_problem(num_of_constraints):
        print("++++++++++++++++++++")
        if optimization_decision == 'minimize':
            print("Minimize the following function f:")
        else:
            print("Maximize the following function f:")
        sp.pprint(obj_function)
        print("Subject to:")
        if num_of_constraints == 1:
            sp.pprint(sp.Eq(constraint[0]))
            print("++++++++++++++++++++")
            l_1 = sp.symbols('l_1', real = True) # creation of lambda symbol(s)
            L = obj_function - l_1* constraint[0] # Lagrangian function 
            gradL = [sp.diff(L,c) for c in [x,y]] # gradient of Lagrangian w.r.t. (x,y)
            KKT_eqs = gradL + [constraint[0]]
            print("The set of equations corresponding to the KKT conditions are:")
            sp.pprint(KKT_eqs)
            print("++++++++++++++++++++")
            stationary_points = sp.solve(KKT_eqs, [x, y, l_1], dict=True) # solve the KKT equations
            print(f'The potential optimizers of f (given all constraint(s) is/are 0) are obtained by solving the KKT equations over x, y, and lambda value(s). Therefore, the constrained value(s) for f are as follows:')
            constrained_values = [obj_function.subs(p) for p in stationary_points]
            if len(constrained_values) == 0:
                print("The function f cannot be optimized given the present constraints.")
            else:
                print(constrained_values)
                print("++++++++++++++++++++")
                if optimization_decision == 'minimize':
                    print(f'The minimum value of f, given the present constraint(s), is {min(constrained_values)}.') 
                else:
                    print(f'The maximum value of f, given the present constraint(s), is {max(constrained_values)}.')
                print("++++++++++++++++++++")
                print("As a result, the values of x, y, and lambda(s) for the constrained value(s) of f are:")
        if num_of_constraints == 2:
            sp.pprint(sp.Eq(constraint[0]))
            sp.pprint(sp.Eq(constraint[1]))
            print("++++++++++++++++++++")
            l_1, l_2 = sp.symbols('l_1 l_2', real = True) # creation of lambda symbol(s)
            L = obj_function - l_1* constraint[0] - l_2*constraint[1] # Lagrangian function
            gradL = [sp.diff(L,c) for c in [x,y]] # gradient of Lagrangian w.r.t. (x,y)
            KKT_eqs = gradL + [constraint[0]] + [constraint[1]]
            print("The set of equations corresponding to the KKT conditions are:")
            sp.pprint(KKT_eqs)
            print("++++++++++++++++++++")
            stationary_points = sp.solve(KKT_eqs, [x, y, l_1, l_2], dict=True) # solve the KKT equations
            print(f'The potential optimizers of f (given all constraint(s) is/are 0) are obtained by solving the KKT equations over x, y, and lambda value(s). Therefore, the constrained value(s) for f are as follows:')
            constrained_values = [obj_function.subs(p) for p in stationary_points]
            if len(constrained_values) == 0:
                print("The function f cannot be optimized given the present constraints.")
            else:
                print(constrained_values)
                print("++++++++++++++++++++")
                if optimization_decision == 'minimize':
                    print(f'The minimum value of f, given the present constraint(s), is {min(constrained_values)}.') 
                else:
                    print(f'The maximum value of f, given the present constraint(s), is {max(constrained_values)}.')
                print("++++++++++++++++++++")
                print("As a result, the values of x, y, and lambda(s) for the constrained value(s) of f are:")
        if num_of_constraints == 3:
            sp.pprint(sp.Eq(constraint[0]))
            sp.pprint(sp.Eq(constraint[1]))
            sp.pprint(sp.Eq(constraint[2]))
            print("++++++++++++++++++++")
            l_1, l_2, l_3 = sp.symbols('l_1 l_2 l_3', real = True) # creation of lambda symbol(s)
            L = obj_function - l_1* constraint[0] - l_2*constraint[1] - l_3*constraint[2] # Lagrangian function
            gradL = [sp.diff(L,c) for c in [x,y]] # gradient of Lagrangian w.r.t. (x,y)
            KKT_eqs = gradL + [constraint[0]] + [constraint[1]] + [constraint[2]]
            print("The set of equations corresponding to the KKT conditions are:")
            sp.pprint(KKT_eqs)
            print("++++++++++++++++++++")
            stationary_points = sp.solve(KKT_eqs, [x, y, l_1, l_2, l_3], dict=True) # solve the KKT equations
            print(f'The potential optimizers of f (given all constraint(s) is/are 0) are obtained by solving the KKT equations over x, y, and lambda value(s). Therefore, the constrained value(s) for f are as follows:')
            constrained_values = [obj_function.subs(p) for p in stationary_points]
            if len(constrained_values) == 0:
                print("The function f cannot be optimized given the present constraints.")
            else:
                print(constrained_values)
                print("++++++++++++++++++++")
                if optimization_decision == 'minimize':
                    print(f'The minimum value of f, given the present constraint(s), is {min(constrained_values)}.') 
                else:
                    print(f'The maximum value of f, given the present constraint(s), is {max(constrained_values)}.')
                print("++++++++++++++++++++")
                print("As a result, the values of x, y, and lambda(s) for the constrained value(s) of f are:")
        if num_of_constraints == 4:
            sp.pprint(sp.Eq(constraint[0]))
            sp.pprint(sp.Eq(constraint[1]))
            sp.pprint(sp.Eq(constraint[2]))
            sp.pprint(sp.Eq(constraint[3]))
            print("++++++++++++++++++++")
            l_1, l_2, l_3, l_4 = sp.symbols('l_1 l_2 l_3 l_4', real = True) # creation of lambda symbol(s)
            L = obj_function - l_1* constraint[0] - l_2*constraint[1] - l_3*constraint[2] - l_4*constraint[3] # Lagrangian function
            gradL = [sp.diff(L,c) for c in [x,y]] # gradient of Lagrangian w.r.t. (x,y)
            KKT_eqs = gradL + [constraint[0]] + [constraint[1]] + [constraint[2]] + [constraint[3]]
            print("The set of equations corresponding to the KKT conditions are:")
            sp.pprint(KKT_eqs)
            print("++++++++++++++++++++")
            stationary_points = sp.solve(KKT_eqs, [x, y, l_1, l_2, l_3, l_4], dict=True) # solve the KKT equations
            print(f'The potential optimizers of f (given all constraint(s) is/are 0) are obtained by solving the KKT equations over x, y, and lambda value(s). Therefore, the constrained value(s) for f are as follows:')
            constrained_values = [obj_function.subs(p) for p in stationary_points]
            if len(constrained_values) == 0:
                print("The function f cannot be optimized given the present constraints.")
            else:
                print(constrained_values)
                print("++++++++++++++++++++")
                if optimization_decision == 'minimize':
                    print(f'The minimum value of f, given the present constraint(s), is {min(constrained_values)}.') 
                else:
                    print(f'The maximum value of f, given the present constraint(s), is {max(constrained_values)}.')
                print("++++++++++++++++++++")
                print("As a result, the values of x, y, and lambda(s) for the constrained value(s) of f are:")
        if num_of_constraints == 5:
            sp.pprint(sp.Eq(constraint[0]))
            sp.pprint(sp.Eq(constraint[1]))
            sp.pprint(sp.Eq(constraint[2]))
            sp.pprint(sp.Eq(constraint[3]))
            sp.pprint(sp.Eq(constraint[4]))
            print("++++++++++++++++++++")
            l_1, l_2, l_3, l_4, l_5 = sp.symbols('l_1 l_2 l_3 l_4 l_5', real = True) # creation of lambda symbol(s)
            L = obj_function - l_1* constraint[0] - l_2*constraint[1] - l_3*constraint[2] - l_4*constraint[3] - l_5*constraint[4] # Lagrangian function
            gradL = [sp.diff(L,c) for c in [x,y]] # gradient of Lagrangian w.r.t. (x,y)
            KKT_eqs = gradL + [constraint[0]] + [constraint[1]] + [constraint[2]] + [constraint[3]] + [constraint[4]]
            print("The set of equations corresponding to the KKT conditions are:")
            sp.pprint(KKT_eqs)
            print("++++++++++++++++++++")
            stationary_points = sp.solve(KKT_eqs, [x, y, l_1, l_2, l_3, l_4], dict=True) # solve the KKT equations
            print(f'The potential optimizers of f (given all constraint(s) is/are 0) are obtained by solving the KKT equations over x, y, and lambda value(s). Therefore, the constrained value(s) for f are as follows:')
            constrained_values = [obj_function.subs(p) for p in stationary_points]
            if len(constrained_values) == 0:
                print("The function f cannot be optimized given the present constraints.")
            else:
                print(constrained_values)
                print("++++++++++++++++++++")
                if optimization_decision == 'minimize':
                    print(f'The minimum value of f, given the present constraint(s), is {min(constrained_values)}.') 
                else:
                    print(f'The maximum value of f, given the present constraint(s), is {max(constrained_values)}.')
                print("++++++++++++++++++++")
                print("As a result, the values of x, y, and lambda(s) for the constrained value(s) of f are:")
        return stationary_points
    opt_solver = solve_problem(number_of_constraints)
    sp.pprint(opt_solver)


# Calling that main method above
if __name__ == '__main__':
    """
    Used to simply call the main method.
    This is a much cleaner approach than using a totally serial approach
    """
    main()