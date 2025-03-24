import numpy as np
import sympy as sp
import plotly.graph_objects as go

# Définition des variables
x, y, u = sp.symbols('x y u')

# Définition de l'EDP : x u_x + (x + y) u_y = u + 1
u_x = sp.Function('u_x')(x, y)
u_y = sp.Function('u_y')(x, y)

# Caractéristiques : dx/dt = x, dy/dt = x + y, du/dt = u + 1
C1, C2 = sp.symbols('C1 C2')
x_char = C1 * sp.exp(y)  # Solution caractéristique pour x
y_char = C1 * (sp.exp(y) - 1)  # Solution caractéristique pour y

# Solution générale
u_general = sp.exp(-y) * (C2 + sp.exp(y) - 1)

# Condition initiale : u(x,0) = x^2
C2_solution = sp.solve(u_general.subs(y, 0) - x**2, C2)[0]

# Expression finale de u(x, y)
u_expr = u_general.subs(C2, C2_solution)

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
