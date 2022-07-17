from dataset import Aug_GlaS

train = Aug_GlaS(train = True)

a = train.__getitem__(2)

print(a[1].shape)

import matplotlib.pyplot as plt

plt.imshow(a[2][0, :, :])
plt.savefig('testing.png')