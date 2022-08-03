import numpy as np
import matplotlib.pyplot as plt

# Mossy layer projection trajectory, 0 deg
projl_MosCA3 = 4
mos_startx0 = np.arange(-20, 21, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0])
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0




projl_MosCA3 = 4
mos_actlen = 10
mos_actstep = 2
num_mosprojs = 8
np.random.seed(2)
mosprojs_angles = np.random.uniform(0, 2*np.pi, num_mosprojs)
np.random.seed(6)
mosprojs_xs = np.random.uniform(-16, 16, num_mosprojs)

mos_startlist, mos_endlist = [], []
for mosproj_angle, mosproj_x in zip(mosprojs_angles, mosprojs_xs):
    mos_v = np.array([np.cos(mosproj_angle), np.sin(mosproj_angle)])
    mos_parmt = np.arange(-mos_actlen/2, mos_actlen/2 + mos_actstep, mos_actstep)
    mos_startvectmp = mos_parmt.reshape(-1, 1) * mos_v.reshape(1, -1)  # (N_steps, 2)
    mos_startvectmp[:, 0 ] = mos_startvectmp[:, 0 ] + mosproj_x
    projvec = projl_MosCA3 * np.array([np.cos(mosproj_angle), np.sin(mosproj_angle)])
    mos_endvectmp = mos_startvectmp + projvec.reshape(1, 2)
    mos_startlist.append(mos_startvectmp)
    mos_endlist.append(mos_endvectmp)
mos_startvec = np.vstack(mos_startlist)
mos_endvec = np.vstack(mos_endlist)

print(mos_startvec.shape)

# Plot the mos conenctions
fig, ax = plt.subplots(figsize=(5, 5))
for mos_start in mos_startlist:

    ax.plot(mos_start[:, 0], mos_start[:, 1])
    ax.plot(mos_start[-1, 0], mos_start[-1, 1], marker='o')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
plt.show()