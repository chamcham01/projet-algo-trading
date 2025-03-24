import numpy as np
import sympy as sp
import plotly.graph_objects as go

# Définition des variables
x, y = sp.symbols('x y')

# Résolution de l'EDP par la méthode des caractéristiques
C = sp.Symbol('C')
solution_general = sp.Eq(C + y, 1/sp.exp(x))

# Exprimer C en fonction de la condition initiale
C_solution = sp.solve(solution_general.subs(y, 0), C)

# Vérification que la solution existe
if not C_solution:
    raise ValueError("Impossible de résoudre l'équation pour C")

C_solution = C_solution[0]

# Expression finale de u(x, y)
u_expr = 1/sp.exp(x) - y

# Convertir en fonction numérique
u_func = sp.lambdify((x, y), u_expr, 'numpy')

# Création des grilles pour x et y
X = np.linspace(-2, 2, 50)
Y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(X, Y)
Z = u_func(X, Y)

# Création de la figure interactive avec Plotly
fig = go.Figure()
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='viridis'))

# Personnalisation du graph
fig.update_layout(
    title="Solution de l'EDP en 3D",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='u(x,y)'
    )
)

# Affichage interactif
fig.show()
