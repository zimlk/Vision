import numpy as np
import matplotlib.pyplot as plt
alpha_x = 557.0943
alpha_y = 712.9824
x0 = 326.3819
y0 = 298.6679
K = np.array([
    [alpha_x, 0, x0, 0],
    [0, alpha_y, y0, 0],
    [0, 0, 1, 0]
])
tx, ty, tz = 100, 0, 1500
rx, ry, rz = 0.8 * np.pi / 2, -1.8 * np.pi / 2, np.pi / 5
def make_projective(ax, ay, az, tx, ty, tz, K):
    rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(ax), -np.sin(ax), 0],
        [0, np.sin(ax), np.cos(ax), 0],
        [0, 0, 0, 1]
    ])
    ry = np.array([
        [np.cos(ay), 0, np.sin(ay), 0],
        [0, 1, 0, 0],
        [-np.sin(ay), 0, np.cos(ay), 0],
        [0, 0, 0, 1]
    ])
    rz = np.array([
        [np.cos(az), -np.sin(az), 0, 0],
        [np.sin(az), np.cos(az), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    R = rx @ ry @ rz
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return K @ R @ T


# Calcul de la matrice de projection initiale
P_initiale = make_projective(rx, ry, rz, tx, ty, tz, K)
# 2. Définition d'un ensemble de points 3D
np.random.seed(50)
X = np.random.randint(-480, 480, size=(3, 6))
# 3. Projection des points 3D sur le plan image
def project(P, X):
    X_h = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0)
    x_h = P @ X_h
    x = x_h / x_h[2, :]
    return x[0:2, :]
x_projected = project(P_initiale, X)
# 4. Visualisation des points projetés en 2D
plt.figure()
plt.scatter(x_projected[0], x_projected[1], color='blue', marker='o')
plt.title("Projection des points 3D en 2D")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.grid()
plt.show()
# 5. Calcul de la matrice de projection avec la méthode DLT
def DLT(p_2D, p_3D):
    Q = []
    for i, j, k, l, m in zip(p_2D[0], p_2D[1], p_3D[0], p_3D[1], p_3D[2]):
        Q.append([k, l, m, 1, 0, 0, 0, 0, -i * k, -i * l, -i * m, -i])
        Q.append([0, 0, 0, 0, k, l, m, 1, -j * k, -j * l, -j * m, -j])
    Q = np.array(Q)
    _, _, Vt = np.linalg.svd(Q)
    A = Vt[-1].reshape(3, 4)
    return A
# Calcul de la matrice de projection avec DLT
P_DLT = DLT(x_projected, X)

#  Comparaison des matrices de projection
# Calcul de l'erreur sans normalisation
erreur_absolue_sans_normalisation = np.abs(P_initiale - P_DLT)
moyenne_erreur_sans_normalisation = np.mean(erreur_absolue_sans_normalisation)
ecart_type_erreur_sans_normalisation = np.std(erreur_absolue_sans_normalisation)

# Normalisation des matrices de projection
P_initiale_normalisee = P_initiale / P_initiale[-1, -1]
P_DLT_normalisee = P_DLT / P_DLT[-1, -1]

# Calcul de l'erreur absolue entre les deux matrices normalisées
erreur_absolue_normalisee = np.abs(P_initiale_normalisee - P_DLT_normalisee)
moyenne_erreur_normalisee = np.mean(erreur_absolue_normalisee)
ecart_type_erreur_normalisee = np.std(erreur_absolue_normalisee)

# 6. Ajoutez du bruit gaussien aux points 2D
x_bruit = x_projected + np.random.normal(0, 1, (2, 6))
P_bruit = DLT(x_bruit, X)
P_bruit_normalisee = P_bruit / P_bruit[-1, -1]
erreur_absolue_bruite = np.abs(P_initiale_normalisee - P_bruit_normalisee)
moyenne_erreur_bruite = np.mean(erreur_absolue_bruite)
ecart_type_erreur_bruite = np.std(erreur_absolue_bruite)
erreur_absolue_bruite_DLT = np.abs(P_DLT_normalisee - P_bruit_normalisee)
moyenne_erreur_bruite_DLT = np.mean(erreur_absolue_bruite_DLT)
ecart_type_erreur_bruite_DLT = np.std(erreur_absolue_bruite_DLT)
# Étape 4 : Visualisation des points originaux et bruités
plt.figure()
plt.scatter(x_projected[0], x_projected[1], color='blue', marker='o', label='Points originaux')
plt.scatter(x_bruit[0], x_bruit[1], marker='+', color='red', label='Points bruités')
plt.title("Projection des points 2D avec et sans bruit")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.legend()
plt.grid()
plt.show()



# 7. Réalisez une nouvelle fonction DTLN
def DLTN(x_projected, X):
    def normalize(points):
        centroid = np.mean(points[:2, :], axis=1)
        points_centered = points[:2, :] - centroid[:, np.newaxis]
        mean_distance = np.mean(np.sqrt(np.sum(points_centered**2, axis=0)))
        scale = np.sqrt(2) / mean_distance
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        points_norm = T @ points
        return points_norm, T
    p_2D_h = np.vstack((x_projected, np.ones(x_projected.shape[1])))
    p_3D_h = np.vstack((X, np.ones(X.shape[1])))
    p_2D_norm, T_2D = normalize(p_2D_h)
    def normalize_3D(points):
        centroid = np.mean(points[:3, :], axis=1)
        points_centered = points[:3, :] - centroid[:, np.newaxis]
        mean_distance = np.mean(np.sqrt(np.sum(points_centered**2, axis=0)))
        scale = np.sqrt(3) / mean_distance  # Échelle pour 3D
        T = np.eye(4)
        T[:3, :3] *= scale
        T[:3, 3] = -scale * centroid
        points_norm = T @ points
        return points_norm, T

    p_3D_norm, T_3D = normalize_3D(p_3D_h)
    A_norm = DLT(p_2D_norm[:2], p_3D_norm[:3])
    A = np.linalg.inv(T_2D) @ A_norm @ T_3D
    return A / A[-1, -1]
# Calcul de la matrice de projection avec DLTN
P_DLTN = DLTN(x_projected, X)
P_DLTN_normalisee = P_DLTN/ P_DLTN[-1, -1]
erreur_absolue_DLTN = np.abs(P_DLTN_normalisee - P_initiale_normalisee)
moyenne_erreur_DLTN = np.mean(erreur_absolue_DLTN)
ecart_type_erreur_DLTN = np.std(erreur_absolue_DLTN)
#8.
x_recalcule_DLT = project(P_DLT, X)
x_recalcule_DLTN = project(P_DLTN, X)

erreur_DLT_rec = np.abs(x_projected - x_recalcule_DLT)
moyenne_erreur_DLT_rec = np.mean(erreur_DLT_rec)
ecart_type_DLT = np.std(erreur_DLT_rec)
erreur_DLTN = np.abs(x_projected - x_recalcule_DLTN)
moyenne_erreur_DLTN_rec = np.mean(erreur_DLTN)
ecart_type_DLTN = np.std(erreur_DLTN)

plt.figure()
plt.scatter(x_projected[0], x_projected[1], color='blue', marker='o', label='Points originaux')
plt.scatter(x_recalcule_DLT[0], x_recalcule_DLT[1], color='green', marker='x', label='Points recalculés (DLT)')
plt.scatter(x_recalcule_DLTN[0], x_recalcule_DLTN[1], color='red', marker='+', label='Points recalculés (DLTN)')
plt.title("Comparaison des points projetés avec DLT et DLTN")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.legend()
plt.grid()
plt.show()

"""Commentaires:
 1. Calcul de la matrice de projection sans normalisation (méthode DLT):
  la méthode DLT sans normalisation est sensible aux différences d'échelle et à la dispersion des points.
 2. Ajout de bruit gaussien et recalcul de la matrice de projection:
  l'ajout du bruit a augmenté l'erreur de projection, ce qui démontre que la méthode DLT est assez sensible aux imperfections dans les données. Comparée à la matrice initiale, l'erreur absolue moyenne a augmenté, ce qui montre que l'algorithme est perturbé par l'ajout de bruit
3.la fonction DLTN avec normalisation:
   Comparée à la méthode DLT sans normalisation, la version DLTN a montré une réduction significative des erreurs de projection, confirmant l'importance de normaliser les points avant d'appliquer DLT. La normalisation rend le calcul plus précis et robuste face aux erreurs numériques.
   4.En recalculant les points 2D à partir des matrices de projection DLT et DLTN, les résultats ont montré que la méthode DLTN produit des projections plus précises. Les points projetés obtenus à partir de la matrice DLTN sont plus proches des points 2D d'origine, ce qui confirme l'efficacité de la normalisation. En comparaison avec les résultats obtenus dans la question 2, les projections DLTN ont montré une amélioration notable de la précision.
    Conclusuion: la méthode DLTN est plus stable et plus précise que DLT, particulièrement dans des situations où les données sont bruitées ou mal conditionnées."""
# 9.
X_8 = np.random.randint(-480, 480, size=(3, 8))
x_8 = project(P_DLT, X_8)
P_DLT8 = DLT(x_8, X_8)
x_8n = project(P_DLTN, X_8)
P_DLTN8= DLTN(x_8n, X_8)
X_50 = np.random.randint(-480, 480, size=(3, 50))
x_50 = project(P_DLT, X_50)
P_DLT50 = DLT(x_50, X_50)
x_50n = project(P_DLTN, X_50)
P_DLTN50= DLTN(x_50n, X_50)
# Calcul des erreurs pour DLT et DLTN avec 8 points
erreur_DLT8 = np.abs(P_DLT8 / P_DLT8[-1, -1] - P_initiale_normalisee)
erreur_DLTN8 = np.abs(P_DLTN8 / P_DLTN8[-1, -1] - P_initiale_normalisee)
# Calcul des erreurs pour DLT et DLTN avec 50 points
erreur_DLT50 = np.abs(P_DLT50 / P_DLT50[-1, -1] - P_initiale_normalisee)
erreur_DLTN50 = np.abs(P_DLTN50 / P_DLTN50[-1, -1] - P_initiale_normalisee)
# Moyennes et écarts-types des erreurs
moyenne_erreur_DLT8 = np.mean(erreur_DLT8)
ecart_type_erreur_DLT8 = np.std(erreur_DLT8)
moyenne_erreur_DLTN8 = np.mean(erreur_DLTN8)
ecart_type_erreur_DLTN8 = np.std(erreur_DLTN8)

moyenne_erreur_DLT50 = np.mean(erreur_DLT50)
ecart_type_erreur_DLT50 = np.std(erreur_DLT50)
moyenne_erreur_DLTN50 = np.mean(erreur_DLTN50)
ecart_type_erreur_DLTN50 = np.std(erreur_DLTN50)
"""En résumé,Avec un nombre croissant de points (8 → 50), la précision des matrices de projection calculées par DLT et DLTN s'améliore
CONCLUSION FINALE:
-DLTN est préférable pour les ensembles de points petits ou bruités.
-Les résultats des deux méthodes convergent lorsque le nombre de points augmente."""