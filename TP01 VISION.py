#1Coordonnées homogènes
#1.1
import numpy as np
droite1 = np.array([1, 1, -5])
droite2 = np.array([4, -5, 7])
# Calcul de l'intersection en homogènes
intersec_homog1 = np.cross(droite1, droite2)
# la méthode avec matrice antisym est dans la QST 3
# vers cartésiennes
def to_cartesian(homogeneous_coords):
    if homogeneous_coords[2] == 0:
        return None  # w=0,
    return homogeneous_coords[:2] / homogeneous_coords[2]

# Calcul de l'intersection en cartésiennes
intersec_cartes2 = to_cartesian(intersec_homog2)
print("Intersection en homogènes :", intersec_homog1)
print("Intersection en cartésiennes :", intersec_cartes1)


# 2.2
droite3 = np.array([1, 2, 1])
droite4 = np.array([3, 6, -2])
intersec_homog2 = np.cross(droite3, droite4)
def to_cartesian(homogeneous_coords):
    if homogeneous_coords[2] == 0:
        return None  # w=0,
    return homogeneous_coords[:2] / homogeneous_coords[2]

print("Intersection en homogènes :", intersec_homog2)
print("Intersection en cartésiennes :", intersec_cartes2)


# 3.3
def matrice_antisymetrique(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

pt1 = np.array([1, 3, 1])
pt2 = np.array([2, 7, 1])
vecteur_directeur = pt2 - pt1
droite_passe_points = matrice_antisymetrique(vecteur_directeur)

print("Équation de la droite en matrice antisymétrique :")
print(droite_passe_points)
#methode 2
pt1 = np.array([1, 3, 1])
pt2 = np.array([2, 7, 1])
droite_passe_points = np.cross(pt1, pt2)
print("Équation de la droite en homogènes :", droite_passe_points)

# 2 Transformations Géométriques
#matrice de transformation en 3D
#11111111111111111111111111111111111
M = np.array([[0.75, -0.43, 0.5, 5],
              [0.65, 0.625, -0.43, 2],
              [-0.125, 0.65, 0.75, 3],
              [0, 0, 0, 1]])
W = M[0:3, 0:3]
Q = M[0:3, 3]

print("W (Matrice de rotation et d'échelle) :\n", W)
print("Q (Vecteur de translation) :\n", Q)
print("M[3, :] (Dernière ligne de la matrice) :\n", M[3, :])

H = np.array([[0, -1, 0, 2],
              [1, 0, 0, 5],
              [0, 0, 1, 4],
              [0, 0, 0, 1]])

A = np.array([3, 4, 5, 1])
A_transformed = H @ A

norm_A_before = np.linalg.norm(A[:-1])
norm_A_after = np.linalg.norm(A_transformed[:-1])

print("Vecteur A (avant transformation) :", A[:-1])
print("Vecteur transformé A :", A_transformed[:-1])
print("Norme de A avant transformation :", norm_A_before)
print("Norme de A après transformation :", norm_A_after)

def rotation_matrix_x(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rotation_matrix_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rotation_matrix_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def scaling_matrix(s):
    return np.array([[s, 0, 0, 0],
                     [0, s, 0, 0],
                     [0, 0, s, 0],
                     [0, 0, 0, 1]])

def transformation_matrix(angles, translation, scale):
    theta_x, theta_y, theta_z = angles
    Rx = rotation_matrix_x(theta_x)
    Ry = rotation_matrix_y(theta_y)
    Rz = rotation_matrix_z(theta_z)
    R = Rz @ Ry @ Rx
    T = translation_matrix(*translation)
    S = scaling_matrix(scale)
    M = T @ R @ S
    return M

points = np.array([[-0.5, -0.5, 0, 1],
                   [-0.5, 0.5, 0, 1],
                   [0.5, -0.5, 0, 1],
                   [0.5, 0.5, 0, 1]])

angles = (np.radians(45), np.radians(0), np.radians(45))
translation = (0.5, 0.5, 2)
scale = 2

M = transformation_matrix(angles, translation, scale)
transformed_points = np.dot(M, points.T).T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Avant Transformation')
ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], color='red', label='Après Transformation')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Transformation des points du carré')
ax.legend()
plt.show()

print("Points avant transformation :")
print(points[:, :-1])
print("Points après transformation :")
print(transformed_points[:, :-1])
