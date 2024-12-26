import numpy as np
import matplotlib.pyplot as plt
import pykitti
import plyfile as ply
dataset = pykitti.raw('C:/Users/Zineb/PycharmProjects/pythonProject3/KITTI_SAMPLE/RAW', '2011_09_26', '0009')
oxts_data = dataset.oxts
plt.figure()
for i in range(len(oxts_data)):
    t = oxts_data[i].T_w_imu[:3, 3]
    plt.scatter(t[0], t[2], c='r')
    plt.scatter(t[1], t[2], c='b')
    plt.scatter(t[1], t[0], c='g')
plt.axis('equal')
plt.show()
points = []
for m in range(50):
    image_cam2 = np.array(dataset.get_cam2(m))
    velo = dataset.get_velo(m)
    velo = velo[velo[:, 0] > 5].T
    velo[3] = 1
    K = dataset.calib.K_cam2
    T = dataset.calib.T_cam2_velo[:3, :4]
    P = K @ T
    x = P @ velo
    x = (x / x[2])[:2]
    mask = (((0 < x[0]) & (x[0] < image_cam2.shape[1] - 1)) & ((0 < x[1]) & (x[1] < image_cam2.shape[0] - 1)))
    x = x[:, mask]
    P_rgb = np.zeros((6, x.shape[1]))
    v = dataset.oxts[m].T_w_imu @ dataset.calib.T_velo_imu @ velo
    P_rgb[:3] = v[:3][:, mask]
    for i in range(x.shape[1]):
        py, px = x[:, i].astype(int)
        P_rgb[3:, i] = image_cam2[px, py]
    points.extend(list(zip(P_rgb[0], P_rgb[1], P_rgb[2], P_rgb[3], P_rgb[4], P_rgb[5])))
vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
el = ply.PlyElement.describe(vertex, 'vertex')
ply.PlyData([el]).write('C:/Users/Zineb/Desktop/BINARY.ply')
