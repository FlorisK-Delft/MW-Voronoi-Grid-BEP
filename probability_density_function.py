import numpy as np

# 1 = gaussian 2d (2 spreaded peaks)
# 2 = gaussian 2d

def pdfunction(x_mesh, y_mesh, type=3, sigma_x=2, sigma_y=2):
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
        return trippel_gaussian(x_mesh, y_mesh, x_center, y_center, sigma_x, sigma_y)



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

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    z_mesh =  np.exp(-10 * (((x - x0) / xsig) ** 2 + ((y - y0) / ysig) ** 2))
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

def trippel_gaussian(x_mesh, y_mesh, x_center, y_center, xsig, ysig):
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
