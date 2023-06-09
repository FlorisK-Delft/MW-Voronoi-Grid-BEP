import numpy as np
import matplotlib.pyplot as plt

# see for types: https://docs.google.com/spreadsheets/d/1mJfya8lhPmmD0_5OJd1RCTz9_rVLU0F96eKJk6eea8g/edit?usp=sharing

def pdfunction(x_mesh, y_mesh, type=3, sigma_x=5, sigma_y=5):
    x_min = np.min(x_mesh)
    x_max = np.max(x_mesh)

    x_center = (x_max - x_min) / 2 + x_min

    y_min = np.min(y_mesh)
    y_max = np.max(y_mesh)

    y_center = (y_max - y_min) / 2 + y_min

    if type == 1:
        return original_gaussian(x_mesh, y_mesh)
    elif type == 2:
        return gaussian_2d(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 3:
        return gaussian_2d_2(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 4:
        return triple_gaussian(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 5:
        return two_different_sized_peaks(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 6:
        return trippel_gaussian_with_diff_speed(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 7:
        return trippel_gaussian_diff_sigma(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 8:
        return gaussian_2d_2_diff_sigma(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 9:
        return quadrant_gaussian(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 10:
        return quadrant_gaussian_diff_height(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)
    elif type == 11:
        return quadrant_gaussian_diff_sigma(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)


def gaussian_2d_2(x_mesh, y_mesh, x0, y0, xsig, ysig):
    xsig_top = xsig
    ysig_top = ysig
    x0_bottom = x0 * 1.5
    y0_bottom = y0 * 0.5

    x0 *= 0.5
    y0 *= 1.5

    xsig_bottom = xsig
    ysig_bottom = ysig

    top_quadrant = np.exp(-8 * (((x_mesh - x0) / xsig_top) ** 2 + ((y_mesh - y0) / ysig_top) ** 2))
    bottom_quadrant = (np.exp(-8 * (((x_mesh - x0_bottom) / xsig_bottom) ** 2
                                    + ((y_mesh - y0_bottom) / ysig_bottom) ** 2)))

    z_mesh = top_quadrant + bottom_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def gaussian_2d_2_diff_sigma(x_mesh, y_mesh, x0, y0, xsig, ysig):
    xsig_top = xsig*1.7
    ysig_top = ysig*1.7
    x0_bottom = x0 * 1.5
    y0_bottom = y0 * 0.5

    x0 *= 0.5
    y0 *= 1.5

    xsig_bottom = xsig*0.7
    ysig_bottom = ysig*0.7

    top_quadrant = np.exp(-8 * (((x_mesh - x0) / xsig_top) ** 2 + ((y_mesh - y0) / ysig_top) ** 2))
    bottom_quadrant = (np.exp(-8 * (((x_mesh - x0_bottom) / xsig_bottom) ** 2
                                    + ((y_mesh - y0_bottom) / ysig_bottom) ** 2)))

    z_mesh = top_quadrant + bottom_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    xsig = 5
    ysig = 5
    z_mesh = np.exp(-10 * (((x - x0) / xsig) ** 2 + ((y - y0) / ysig) ** 2))
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh


def original_gaussian(x_mesh, y_mesh):
    xsig = 5.0
    ysig = 5.0

    x0 = 5.0
    y0 = 5.0

    xsig_top = 0.5 * xsig
    ysig_top = 1.5 * ysig
    x0 *= 0.5
    y0 *= 1.5
    x0_bottom = x0 * 1.5
    y0_bottom = y0 * 0.5
    xsig_bottom = 1.5 * xsig
    ysig_bottom = 0.5 * ysig

    top_quadrant = np.exp(-8 * (((x_mesh - x0) / xsig_top) ** 2 + ((y_mesh - y0) / ysig_top) ** 2))
    bottom_quadrant = (np.exp(-8 * (((x_mesh - x0_bottom) / xsig_bottom) ** 2
                                    + ((y_mesh - y0_bottom) / ysig_bottom) ** 2)))

    z_mesh = top_quadrant + bottom_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh


def triple_gaussian(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center
    center_y_3 = y_center * 0.5

    first_quadrant = np.exp(-8 * (((x_mesh - center_x_1) / xsig) ** 2 + ((y_mesh - center_y_1) / ysig) ** 2))
    second_quadrant = np.exp(-8 * (((x_mesh - center_x_2) / xsig) ** 2 + ((y_mesh - center_y_2) / ysig) ** 2))
    third_quadrant = np.exp(-8 * (((x_mesh - center_x_3) / xsig) ** 2 + ((y_mesh - center_y_3) / ysig) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def two_different_sized_peaks(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 0.5

    first_quadrant = 2*np.exp(-8 * (((x_mesh - center_x_1) / xsig) ** 2 + ((y_mesh - center_y_1) / ysig) ** 2))
    second_quadrant = np.exp(-8 * (((x_mesh - center_x_2) / xsig) ** 2 + ((y_mesh - center_y_2) / ysig) ** 2))

    z_mesh = first_quadrant + second_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def trippel_gaussian_with_diff_speed(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center
    center_y_3 = y_center * 0.5

    first_quadrant = 2.5 * np.exp(-8 * (((x_mesh - center_x_1) / xsig) ** 2 + ((y_mesh - center_y_1) / ysig) ** 2))
    second_quadrant = 1 * np.exp(-8 * (((x_mesh - center_x_2) / xsig) ** 2 + ((y_mesh - center_y_2) / ysig) ** 2))
    third_quadrant = 1.75 * np.exp(-8 * (((x_mesh - center_x_3) / xsig) ** 2 + ((y_mesh - center_y_3) / ysig) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def trippel_gaussian_diff_sigma(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center
    center_y_3 = y_center * 0.5

    first_quadrant = np.exp(-8 * (((x_mesh - center_x_1) / xsig*3) ** 2 + ((y_mesh - center_y_1) / ysig*3) ** 2))
    second_quadrant = np.exp(-8 * (((x_mesh - center_x_2) / xsig*1.5) ** 2 + ((y_mesh - center_y_2) / ysig*1.5) ** 2))
    third_quadrant = np.exp(-8 * (((x_mesh - center_x_3) / xsig*0.8) ** 2 + ((y_mesh - center_y_3) / ysig*0.8) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def quadrant_gaussian(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center * 0.5
    center_y_3 = y_center * 0.5

    center_x_4 = x_center * 1.5
    center_y_4 = y_center * 0.5

    first_quadrant = np.exp(-8 * (((x_mesh - center_x_1) / xsig) ** 2 + ((y_mesh - center_y_1) / ysig) ** 2))
    second_quadrant = np.exp(-8 * (((x_mesh - center_x_2) / xsig) ** 2 + ((y_mesh - center_y_2) / ysig) ** 2))
    third_quadrant = np.exp(-8 * (((x_mesh - center_x_3) / xsig) ** 2 + ((y_mesh - center_y_3) / ysig) ** 2))
    fourth_quadrant = np.exp(-8 * (((x_mesh - center_x_4) / xsig) ** 2 + ((y_mesh - center_y_4) / ysig) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant + fourth_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def quadrant_gaussian_diff_height(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center * 0.5
    center_y_3 = y_center * 0.5

    center_x_4 = x_center * 1.5
    center_y_4 = y_center * 0.5

    first_quadrant = 2*np.exp(-8 * (((x_mesh - center_x_1) / xsig) ** 2 + ((y_mesh - center_y_1) / ysig) ** 2))
    second_quadrant = 1*np.exp(-8 * (((x_mesh - center_x_2) / xsig) ** 2 + ((y_mesh - center_y_2) / ysig) ** 2))
    third_quadrant = 0.8*np.exp(-8 * (((x_mesh - center_x_3) / xsig) ** 2 + ((y_mesh - center_y_3) / ysig) ** 2))
    fourth_quadrant = 1.75*np.exp(-8 * (((x_mesh - center_x_4) / xsig) ** 2 + ((y_mesh - center_y_4) / ysig) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant + fourth_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh

def quadrant_gaussian_diff_sigma(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
    center_x_1 = x_center * 0.5
    center_y_1 = y_center * 1.5

    center_x_2 = x_center * 1.5
    center_y_2 = y_center * 1.5

    center_x_3 = x_center * 0.5
    center_y_3 = y_center * 0.5

    center_x_4 = x_center * 1.5
    center_y_4 = y_center * 0.5

    first_quadrant = np.exp(-8 * (((x_mesh - center_x_1) / xsig*1.3) ** 2 + ((y_mesh - center_y_1) / ysig*1.3) ** 2))
    second_quadrant = np.exp(-8 * (((x_mesh - center_x_2) / xsig*3) ** 2 + ((y_mesh - center_y_2) / ysig*3) ** 2))
    third_quadrant = np.exp(-8 * (((x_mesh - center_x_3) / xsig*1.7) ** 2 + ((y_mesh - center_y_3) / ysig*1.7) ** 2))
    fourth_quadrant = np.exp(-8 * (((x_mesh - center_x_4) / xsig*0.8) ** 2 + ((y_mesh - center_y_4) / ysig*0.8) ** 2))

    z_mesh = first_quadrant + second_quadrant + third_quadrant + fourth_quadrant
    z_mesh /= z_mesh.sum()  # normalize z so the total is equal to 1
    return z_mesh






plt.figure()

resolution_x = (10 * 20 + 1)
resolution_y = (10 * 20 + 1)
xx, yy = np.meshgrid(
    np.linspace(0, 10, resolution_x),
    np.linspace(0, 10, resolution_y),
    )
z = pdfunction(xx, yy, type = 9)

plt.imshow(z, origin='lower',
           extent=(0, 10, 0, 10),
           alpha=0.5)
plt.colorbar()
plt.show()