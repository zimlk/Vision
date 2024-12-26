import numpy as np
import matplotlib.pyplot as plt
# J'ai utilisé cette fonction pour créer une matrice de projection
# qui combine les rotations, la translation, et les paramètres de la caméra.
# Cette matrice de projection va permettre de transformer des points 3D en points 2D.
def make_projective(ax, ay, az, tx, ty, tz, alpha_x, alpha_y, x0, y0):
    # J'ai utilisé la matrice `rx` pour représenter la rotation autour de l'axe x.
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(ax), -np.sin(ax), 0],
                   [0, np.sin(ax), np.cos(ax), 0],
                   [0, 0, 0, 1]])

    # J'ai utilisé la matrice `ry` pour représenter la rotation autour de l'axe y.
    ry = np.array([[np.cos(ay), 0, np.sin(ay), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ay), 0, np.cos(ay), 0],
                   [0, 0, 0, 1]])

    # J'ai utilisé la matrice `rz` pour représenter la rotation autour de l'axe z.
    rz = np.array([[np.cos(az), -np.sin(az), 0, 0],
                   [np.sin(az), np.cos(az), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    #combiné les trois matrices de rotation pour obtenir une matrice de rotation totale `R`.
    R = rx @ ry @ rz

    # la matrice `T` pour ajouter une translation dans l'espace 3D,
    # décalant les points en fonction des valeurs `tx`, `ty` et `tz`.
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])

    #la matrice `K` pour représenter les paramètres intrinsèques de la caméra.
    # Cette matrice convertit les coordonnées 3D en coordonnées 2D en pixels.
    K = np.array([
        [alpha_x, 0, x0, 0],
        [0, alpha_y, y0, 0],
        [0, 0, 1, 0]
    ])

    # J'ai multiplié `K`, `R` et `T` pour obtenir la matrice de projection finale,
    # qui combine la rotation, la translation, et les paramètres de la caméra.
    return K @ R @ T


# J'ai utilisé cette fonction de projection pour transformer des points 3D en points 2D.
# Elle applique la matrice de projection `P` sur les points 3D `X`.
def project(P, X):
    # J'ai ajouté une ligne de 1 pour chaque point pour passer en coordonnées homogènes,
    # ce qui permet de faire la projection 3D en 2D.
    px = P @ np.concatenate((X, np.ones((1, np.shape(X)[1]))), axis=0)

    #normalisé les points en divisant par la troisième ligne pour obtenir les coordonnées en pixels.
    x = px / px[2, :]

    #extrait les deux premières lignes pour obtenir les coordonnées x et y des points projetés en 2D.
    x = x[0:2, :]
    return x


#les paramètres de test pour `make_projective` et `project`
alpha_x = 800
alpha_y = 800
x0 = 250
y0 = 250
thetax = np.radians(20)
Trz = 2

#les coordonnées des sommets d'un cube en 3D
cube = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
]).T

# J'ai utilisé la fonction `make_projective` pour calculer la matrice de projection
# en utilisant les paramètres spécifiés.
M = make_projective(thetax, 0, 0, 0, 0, Trz, alpha_x, alpha_y, x0, y0)

# J'ai appliqué la fonction `project` pour obtenir les coordonnées 2D des sommets du cube.
xxx = project(M, cube)

# J'ai affiché le cube en 3D pour visualiser sa forme avant la projection en 2D.
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter3D(cube[0, :], cube[1, :], cube[2, :], color="green")
plt.title("La projection 3D du CUBE")
plt.show()

# J'ai affiché la projection 2D des sommets du cube pour visualiser le résultat de la projection.
plt.figure()
plt.scatter(xxx[0], xxx[1])
plt.title("Projection 2D du CUBE")
plt.xlabel("x  (pixel)")
plt.ylabel("y  (pixel)")
plt.show()


# Ici, je vais définir les valeurs d'alpha_x, alpha_y, x0, et y0 nécessaires pour l'appel
#Example 1
#alpha_x = 600
#alpha_y = 800
#x0 = 312.6798
#y0 = 266.9831
#Example 2
#alpha_x = 400
#alpha_y = 700
#x0 = 300
#y0 = 200
#Example par des valeurs approximatives typiques que l'on retrouve dans de nombreux caméras
#alpha_x = 800
#alpha_y = 800
#x0 = 320
#y0 = 220 pour une image 640x480 et x0 =640 , y0 = 360 pour une image de résolution plus haute comme 1280x720.
# Correction de l'appel à la fonction `make_projective`
P = make_projective(0.8 * np.pi / 2, -1.9 * np.pi / 2, np.pi / 5, 100, 0, 1500, alpha_x, alpha_y, x0, y0)
#P = make_projective(0.6 * np.pi / 2, -1.2 * np.pi / 2, np.pi / 5, 80, 0, 1000, alpha_x, alpha_y, x0, y0)

#Ici je code les matrices des parametre extrinseques et intrinseques

#les point dans la plage -500 et 500 dans une matrice de 3x6
X = np.random.randint(-500, 500, size=(3, 6))
#utilisant la fonction de projection on a calculer la nouvelle projection des points randomisées
x = project(P, X)
plt.figure()
plt.scatter(x[0], x[1])
plt.show()
#Dans le cadre de ce TP,
# j'ai observé que les résultats de la projection changent lorsque l'on modifie les paramètres intrinsèques de la caméra, à savoir les longueurs focales (\( \alpha_x \) et \( \alpha_y \))
# ainsi que les coordonnées du centre de l'image (\( x_0 \) et \( y_0 \)).
# En particulier, en réduisant les valeurs des focales, la taille des objets projetés diminue, donnant l'impression que les objets sont plus éloignés ou moins détaillés.
# À l'inverse, en augmentant ces valeurs, les objets semblent plus proches ou plus grands.
# De plus, en ajustant les paramètres \( x_0 \) et \( y_0 \), le centre de l'image est déplacé sur l'écran, ce qui entraîne un décalage des points projetés dans le plan 2D.
# Ces ajustements permettent de simuler différentes configurations de caméra et d'observer l'impact de la focale ou du centre de projection sur le rendu final, offrant ainsi un contrôle accru sur l'effet visuel de la projection, comme dans le cas d'une simulation de zoom avant ou arrière.
# Paramètres de base
alpha_x_base = 800
alpha_y_base = 800
x0_base = 250
y0_base = 250
ax = np.radians(20)
ay = 0
az = 0
tx = 0
ty = 0
tz = 2

# Liste de valeurs pour tester chaque paramètre
x0_values = [200, 300, 500, 800, 1000]
y0_values = [200, 300, 500, 800, 1000]
alpha_x_values = [600, 700, 800, 900, 1000]
alpha_y_values = [600, 700, 800, 900, 1000]

# Fonction pour afficher les projections pour une liste de valeurs de paramètre
def plot_projections(param_name, param_values, base_params):
    plt.figure(figsize=(12, 6))
    for i, param_value in enumerate(param_values):
        params = base_params.copy()
        params[param_name] = param_value
        P = make_projective(params['ax'], params['ay'], params['az'], params['tx'], params['ty'], params['tz'], params['alpha_x'], params['alpha_y'], params['x0'], params['y0'])
        projection = project(P, cube)
        plt.subplot(1, len(param_values), i + 1)
        plt.scatter(projection[0], projection[1])
        plt.title(f"{param_name} = {param_value}")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.axis("equal")

    plt.suptitle(f"Effet de la variation de {param_name} sur la projection")
    plt.show()
base_params = {'ax': ax, 'ay': ay, 'az': az, 'tx': tx, 'ty': ty, 'tz': tz, 'alpha_x': alpha_x_base, 'alpha_y': alpha_y_base, 'x0': x0_base, 'y0': y0_base}
plot_projections('x0', x0_values, base_params)
plot_projections('y0', y0_values, base_params)
plot_projections('alpha_x', alpha_x_values, base_params)
plot_projections('alpha_y', alpha_y_values, base_params)
#En variant x0 ou y0, on voir un décalage horizontal ou vertical de l’image projetée.
#En modifiant alpha_x et alpha_y, la taille de l’objet projeté change, simulant un zoom avant ou arrière.
"""Résumé des résultats :
-Paramètres extrinsèques (rotation et translation) ont modifié l'orientation et la position de la caméra, affectant la perspective de l'image projetée
-Paramètres intrinsèques tels que la focale alpha_x et alpha_y et centre de l'image(x_o,y_o) ont permis de simuler des phénomènes comme le zoom avant/arrière et le déplacement de l'image sur le plan 2D. Nous avons observé que la modification de 
   (x_o,y_o) déplaçait respectivement l'image horizontalement et verticalement, tandis que la modification de alpha_x et alpha_y a influencé l'échelle de l'objet projeté.  -"""