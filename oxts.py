import numpy as np
import glob
import matplotlib.pyplot as plt


kitti_path = '../remote_kitti'
oxts_files = glob.glob(kitti_path + '/*/*/oxts/data/*.txt')

print "reading files"
all_data = []

for oxt in oxts_files:
    file = open(oxt)
    data = file.read().split()
    all_data.append(data)

all_data = np.array(all_data, dtype=np.float64)

# Velocities
# vf = oxts[8]    # FORWARD VELOCITY [m/s]
# vl = oxts[9]    # LEFTWARDS VELOCITY
# vu = oxts[10]   # UP VELOCITY

# wf = oxts[20]   # FORWARD AXIS ROTATION
# wl = oxts[21]   # LEFTWARDS AXIS ROTATION
# wu = oxts[22]   # UPWARDS AXIS ROTATION

# to calculate variance, we are going to assume that the distribution is symmetrical
# because we'll be flipping the order of the images with even probability during training
# this means 0 mean.


# forward velocity:
vf = all_data[:,8]
plt.hist(vf)
plt.title('forward velocity')
plt.show()

vf_variance = np.mean(vf**2)
print "VF_VARIANCE = %f" % vf_variance

plt.hist(vf/(vf_variance**0.5), normed=True)
plt.show()

# leftwards velocity:
vl = all_data[:, 9]
plt.hist(vl)
plt.title("leftwards velocity")
plt.show()

vl_variance = np.mean(vl**2)
print "VL_VARIANCE = %f" % vl_variance

# up velocity:
vu = all_data[:, 10]
plt.hist(vu)
plt.title("upwards velocity")
plt.show()

vu_variance = np.mean(vu**2)
print "VU_VARIANCE = %f" % vu_variance

# forward axis rotation:
vf = all_data[:, 20]
plt.hist(vf)
plt.title('forward axis rotation')
plt.show()

vf_variance = np.mean(vf**2)
print "WF_VARIANCE = %f" % vf_variance

# leftwards axis rotation:
vl = all_data[:, 21]
plt.hist(vl)
plt.title("leftwards axis rotation")
plt.show()

vl_variance = np.mean(vl**2)
print "WL_VARIANCE = %f" % vl_variance

# up axis rotation:
vu = all_data[:, 22]
plt.hist(vu)
plt.title("upwards axis rotation")
plt.show()

vu_variance = np.mean(vu**2)
print "WU_VARIANCE = %f" % vu_variance
