import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Load your fonts
font_path_regular = './lmroman7-regular.otf'
font_prop_regular = fm.FontProperties(fname=font_path_regular)

font_path_bold = './lmroman7-bold.otf'
font_prop_bold = fm.FontProperties(fname=font_path_bold)

def print_grid(ax, grid_mesh, title):
    ax.imshow(grid_mesh, cmap='binary', origin='upper', extent=[0, len(grid_mesh[0]), 0, len(grid_mesh)])

    # Loop over data dimensions and create text annotations.
    for i in range(len(grid_mesh)):
        for j in range(len(grid_mesh[i])):
            text = ax.text(j + 0.5, len(grid_mesh) - i - 0.5, grid_mesh[i, j],
                           ha="center", va="center", color="r", fontsize=16,
                           fontproperties=font_prop_bold)  # use the bold font for the numbers

    ax.set_xticks([])  # Remove x-axis values
    ax.set_yticks([])  # Remove y-axis values
    ax.set_title(title, fontproperties=font_prop_regular, fontsize = 25)  # use the regular font for the title



# Creating the array
grid = np.array([
    [1, 1, 1, 1, 1, 3, 3, 3, 3],
    [1, 1, 1, 1, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2, 2, 2],
    [1, 1, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 2, 2, 2, 2, 2, 2, 2]
])

# calculate what the gradient is and if any gradient exist make it equal to 1, otherwise make 'almost' 0 = 0
grad = np.gradient(grid)
grad = np.where(np.isclose(grad, 0, atol=1e-8), 0, 1)

# adds the gradient in y and in x direction together so a border forms when plotted
border = grad[0] + grad[1]

# makes sure all values are either 0 or 1 so the border can be plotted with a homogeneous colour
border = np.where(np.isclose(border, 0, atol=1e-8), 0, 1)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

print_grid(axs[0], grid, 'Original Grid')
print_grid(axs[1], border, 'Border')

width_arrow = 0.06
# Add arrow
arrow = plt.Arrow(0.5 - 0.3*width_arrow, 0.5, width_arrow, 0,
                  width=0.1, color='black', transform=fig.transFigure, figure=fig)
fig.patches.extend([arrow])

plt.savefig("visualise_mw_voronoi.png", dpi=200)  # dpi is the resolution of each png
plt.show()
