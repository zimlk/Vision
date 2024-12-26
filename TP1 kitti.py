import matplotlib.pyplot as plt
import numpy as np
import pykitti
np.set_printoptions(suppress=True, precision=6, edgeitems=30, linewidth=120)
dataset = pykitti.raw('C:/Users/Zineb/PycharmProjects/pythonProject3/KITTI_SAMPLE/RAW', '2011_09_26', '0009', frames=range(0, 5, 1))
image_cam2 = dataset.get_cam2(0)
velo = dataset.get_velo(0)
velo = velo[velo[:, 0] >5]
T_p = dataset.calib.T_cam2_velo
K = dataset.calib.K_cam2
K = np.hstack((K, np.atleast_2d(np.array([0, 0, 0])).T))
K = np.vstack([K, ([0, 0, 0, 1])])
T = K @ T_p
velo.T[3, :] = np.ones((1, velo.shape[0]))
P_cam2 = T @ velo.T
P_cam2_to_c = P_cam2[0:2, :] / P_cam2[2, :]
mask = (P_cam2_to_c[0,:]>= 0) * ((P_cam2_to_c[0, :] < np.shape(image_cam2)[1])  & (P_cam2_to_c[1, :] < np.shape(image_cam2)[0]))
P_cam2_to_h_x = P_cam2_to_c[0, :][mask.ravel() == True]
P_cam2_to_h_y = P_cam2_to_c[1, :][mask.ravel() == True]
P_cam2_to_h_z = velo[:, 0][mask.ravel() == True]
P_cam2_to_h_z = 1 / P_cam2_to_h_z
plt.figure()
plt.imshow(image_cam2)
plt.scatter(x=P_cam2_to_h_x, y=P_cam2_to_h_y,s=1, c=P_cam2_to_h_z, cmap="jet")
plt.show()
