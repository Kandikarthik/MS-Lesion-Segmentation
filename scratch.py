import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# A typical Mendeley shape e.g. 512, 512, 32
image = np.zeros((512, 512, 32))
s_x, s_y, s_z = image.shape
aspect_sagittal = s_y / (s_z + 1e-5)
aspect_coronal = s_x / (s_z + 1e-5)
aspect_axial = s_x / (s_y + 1e-5)

axes[0].imshow(np.rot90(image[256, :, :]), aspect=aspect_sagittal)
axes[0].set_title('Sagittal (x=240)')

axes[1].imshow(np.rot90(image[:, 256, :]), aspect=aspect_coronal)
axes[1].set_title('Coronal')

axes[2].imshow(np.rot90(image[:, :, 16]), aspect=aspect_axial)
axes[2].set_title('Axial')

fig.tight_layout()
fig.savefig('test_plot_1.png')
print("Done 1")
