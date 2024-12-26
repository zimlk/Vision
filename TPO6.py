import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
R02 = cv2.imread("C:/Users/Zineb/Desktop/M2 TSI IMOVI/vision/R02.png", cv2.IMREAD_GRAYSCALE)
L01 = cv2.imread("C:/Users/Zineb/Desktop/M2 TSI IMOVI/vision/L01.png", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp1, ds1 = sift.detectAndCompute(R02, None)
kp2, ds2 = sift.detectAndCompute(L01, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(ds1, ds2)
random.seed(42)
sample_matches = random.sample(matches, 50)
img_matches = cv2.drawMatches(R02, kp1, L01, kp2, sample_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.show()
pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.01)
inliers = mask.ravel() == 1
pts1_inliers = pts1[inliers]
pts2_inliers = pts2[inliers]
inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
sample_inliers = random.sample(inlier_matches, min(50, len(inlier_matches)))
img_matches_inliers = cv2.drawMatches(R02, kp1, L01, kp2, sample_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_inliers)
plt.title("50 Points homologues Inliers (corrects)")
plt.show()
outliers = mask.ravel() == 0
pts1_outliers = pts1[outliers]
pts2_outliers = pts2[outliers]
outlier_matches = [matches[i] for i in range(len(matches)) if not inliers[i]]
sample_outliers = random.sample(outlier_matches, min(50, len(outlier_matches)))
img_matches_outliers = cv2.drawMatches(R02, kp1, L01, kp2, sample_outliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_outliers)
plt.title("50 Points homologues Outliers (incorrects)")
plt.show()
inliers_errors = []
outliers_errors = []
for pt1, pt2 in zip(pts1[inliers], pts2[inliers]):
    x1 = np.array([pt1[0], pt1[1], 1]).reshape(-1, 1)
    x2 = np.array([pt2[0], pt2[1], 1]).reshape(1, -1)
    error = np.abs(x2 @ F @ x1)[0][0]
    inliers_errors.append(error)
for pt1, pt2 in zip(pts1[outliers], pts2[outliers]):
    x1 = np.array([pt1[0], pt1[1], 1]).reshape(-1, 1)
    x2 = np.array([pt2[0], pt2[1], 1]).reshape(1, -1)
    error = np.abs(x2 @ F @ x1)[0][0]
    outliers_errors.append(error)
inliers_mean = np.mean(inliers_errors)
inliers_std = np.std(inliers_errors)
outliers_mean = np.mean(outliers_errors)
outliers_std = np.std(outliers_errors)
print("\nErreurs pour les inliers:")
print(f"Moyenne: {inliers_mean:.4f}, Ecart type: {inliers_std:.4f}")
print("\nErreurs pour les outliers:")
print(f"Moyenne: {outliers_mean:.4f}, Ecart type: {outliers_std:.4f}")

#commentaires
"""
Commentaires :
1. Les correspondances aléatoires (4) peuvent inclure des points incorrects (outliers).
2. Après RANSAC (6), seules les correspondances inliers correctes sont affichées, 
   montrant une meilleure cohérence avec la géométrie épipolaire.
3. Les erreurs moyennes des inliers (8) sont bien plus faibles que celles des outliers, 
   ce qui confirme la robustesse de RANSAC dans la réjection des outliers.
"""
