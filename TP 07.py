import cv2
import numpy as np
import matplotlib.pyplot as plt
import pykitti
import os

np.set_printoptions(suppress=True, precision=6, edgeitems=30, linewidth=120)
dataset = pykitti.raw('C:/Users/Zineb/PycharmProjects/pythonProject3/KITTI_SAMPLE/RAW', '2011_09_26', '0009',
                      frames=range(11))
K = dataset.calib.K_cam2
output_dir = r"C:\Users\Zineb\Desktop\M2 TSI IMOVI\images"
os.makedirs(output_dir, exist_ok=True)
couples = [[1, 2], [3, 4], [5, 8], [7, 10], [6, 10]]
for cp1, cp2 in couples:
    img1 = np.array(dataset.get_cam2(cp1))
    img2 = np.array(dataset.get_cam3(cp2))
    sift = cv2.SIFT_create()
    kp1, ds1 = sift.detectAndCompute(img1, None)
    kp2, ds2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(ds1, ds2)
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inlier_pts1 = np.array([pts1[j] for j in range(len(pts1)) if mask[j]])
    inlier_pts2 = np.array([pts2[j] for j in range(len(pts2)) if mask[j]])
    _, R, T, _, pts_3d_h = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K, distanceThresh=100)
    pts_3d = pts_3d_h / pts_3d_h[3, :]
    Z = pts_3d[2, :]
    mask_pts = (Z >= np.percentile(Z, 5)) & (Z <= np.percentile(Z, 95))
    filtered_pts_3d = pts_3d[:, mask_pts]
    filtered_pts1 = inlier_pts1[mask_pts]
    filtered_pts2 = inlier_pts2[mask_pts]
    Z = filtered_pts_3d[2, :]
    plt.figure()
    plt.imshow(img1), plt.axis('off')
    plt.scatter(filtered_pts1[:, 0], filtered_pts1[:, 1], c=Z, s=7, cmap='jet_r')
    plt.title(f"Image {cp1} (cam2) avec points projetés")
    plt.savefig(os.path.join(output_dir, f"image_{cp1}_cam2_points.png"))
    plt.figure()
    plt.imshow(img2), plt.axis('off')
    plt.scatter(filtered_pts2[:, 0], filtered_pts2[:, 1], c=Z, s=7, cmap='jet_r')
    plt.title(f"Image {cp2} (cam3) avec points projetés")
    plt.savefig(os.path.join(output_dir, f"image_{cp2}_cam3_points.png"))

plt.show()
