import configparser
import datetime
import json
import logging
import os
import time

import cv2
import numpy as np
import scipy as sp
import scipy.interpolate as interp
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import skfmm
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial import Voronoi, voronoi_plot_2d

import planebots
from planebots import packdir, log_short
from planebots.control import dwa

# Initialize saving directory:

suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
savedir = os.path.join(planebots.packdir, "data", "coverage_simulation", suffix)

savename = os.path.join(savedir, "dataArrays.npz")

ini_filepath = os.path.join(packdir, 'settings', 'coverage.ini')
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read(ini_filepath)
# Load all settings:
domain_mm = json.loads(config.get("coverage", "domain_mm"))
gradient_epsilon = json.loads(config.get("coverage", "gradient_epsilon"))
grid_size = json.loads(config.get("coverage", "grid_size"))
lloyd_max_iterations = json.loads(config.get("coverage", "lloyd_max_iterations"))
n_agents = json.loads(config.get("coverage", "n_agents"))
lloyd_max_error = json.loads(config.get("coverage", "lloyd_max_error"))
normalized_positions = json.loads(config.get("coverage", "normalized_positions"))
max_num_goals = json.loads(config.get("coverage", "max_num_goals"))
max_path_points = json.loads(config.get("coverage", "max_path_points"))
max_spline_points = json.loads(config.get("coverage", "max_spline_points"))
production_multiplier = json.loads(config.get("coverage", "production_multiplier"))
ignore_dwa = config.getboolean("coverage", "ignore_dwa")
length_mp = json.loads(config.get("coverage", "length_mp"))
screenshots = config.getboolean("coverage", "screenshots")
decay = json.loads(config.get("coverage", "decay"))
screenshots_from = json.loads(config.get("coverage", "screenshots_from"))
max_goal_switches = json.loads(config.get("coverage", "max_goal_switches"))
aspect = json.loads(config.get("coverage", "aspect"))
cover_radius = json.loads(config.get("coverage", "cover_radius"))
z_star = json.loads(config.get("coverage", "z_star"))
debug = config.getboolean("coverage", "debug")
update_voronoi = config.getboolean("coverage", "update_voronoi")
z_star = json.loads(config.get("coverage", "z_star"))
dynamic_rho = config.getboolean("coverage", "dynamic_rho")
rho_max = json.loads(config.get("coverage", "rho_max"))
fmm_test_point = json.loads(config.get("fmm", "test_point"))
path_step_size = json.loads(config.get("fmm", "path_step_size"))
beta = json.loads(config.get("fmm", "beta"))
fmm_end_point = json.loads(config.get("fmm", "test_end_point"))
c_border = json.loads(config.get("fmm", "c_border"))
max_plot_trajectory_points = json.loads(config.get("plotting", "max_plot_trajectory_points"))
goal_radius_prox = json.loads(config.get("coverage", "goal_radius_prox"))
end_plot = config.getboolean("plotting", "end_plot")
show_grid_points = config.getboolean("plotting", "show_grid_points")
show_improvement = config.getboolean("plotting", "show_improvement")
show_fmm_grids = config.getboolean("plotting", "show_fmm_grids")
show_coverage_field = config.getboolean("plotting", "show_coverage_field")
show_grid_lines = config.getboolean("plotting", "show_grid_lines")
show_optimal_paths = config.getboolean("plotting", "show_optimal_paths")
show_dw = config.getboolean("plotting", "show_dw")
show_init_points = config.getboolean("plotting", "show_init_points")
show_velo_field = config.getboolean("plotting", "show_velo_field")
show_goal_text = config.getboolean("plotting", "show_goal_text")

# Init data with config values
n_iter = int(dwa.config.t_sim / dwa.config.dt)
a = aspect  # dims are m x n = d-a,d+a
gsx, gsy = grid_size - a, grid_size + a  # Dimensions of the grid
goals = np.ones((n_agents, max_num_goals, 2)) * np.nan  # Contains the goals
goals_coords = np.ones((n_agents, max_num_goals, 2)) * np.nan  # Contains the goals
goals_coords_max = np.ones((n_agents, 2)) * np.nan  # Contains the goals
goals_index = np.zeros((n_agents), np.uint16)  # Contains where the agents are in the trajectory to the goal
goals_max_index = np.zeros((n_agents), np.uint16)  # Index of the goal with the highest score
goals_scores = np.ones((n_agents, max_num_goals)) * np.nan  # Contains the scores
goal_paths = np.ones((n_agents, max_num_goals, max_spline_points,
                      2)) * np.nan  # Contains the paths to goals, shortened with spline interpolation
goal_paths_org = np.ones(
    (n_agents, max_num_goals, max_path_points, 2)) * np.nan  # Contains the paths obtained with gradient descent
FMMgrids = np.ones((n_agents, gsy, gsx)) * np.nan  # Contains time to agent values for the grid
FMMgradients = np.ones(
    (n_agents, gsy, gsx, 2)) * np.nan  # Contains derivatives of the time to agent values for the grid
FMMcombined = np.zeros((gsy, gsx))  # Contains combined time to agent values for the grid
F0 = np.zeros((gsy, gsx))  # Contains combined time to agent values for the grid
Ftot = np.zeros((gsy, gsx))  # Contains combined time to agent values for the grid
states_all = np.ones((n_agents, 5)) * 0
states_all_traj = np.ones((n_agents, n_iter, 5)) * np.nan  # Trajectory travelled by agents

## Data to save:
# iter_Z = np.zeros((n_iter,gsy, gsx)) * np.nan
# iter_goal_paths_org = np.ones((n_iter,n_agents, max_num_goals, max_path_points, 2)) * np.nan


Z = np.zeros((gsy, gsx))  # Contains the coverage level
Alpha_i = np.zeros((gsy, gsx)) * np.nan  # Agent coverage fcn discretized over grid
Zstart = np.ones((gsy, gsx))  # Initial coverage level
qs = np.ones((gsy * gsx, 2)) * np.nan  # Holds the coordinates
qsr = qs.view()  # Holds the coordinates in 2d, underlying data is the same as qs
qsr.shape = (gsy, gsx, 2)
k = np.ones((gsy, gsx), np.int8) * -1
kv = k.view()  # Holds the coordinates in 2d, underlying data is the same as qs
kv.shape = (gsy * gsx)
M_impr = np.zeros((gsy, gsx), np.float32)

M_impr_delta = np.zeros((gsy, gsx), np.float32)  # Delta will not be used

gridlines1 = gly = np.linspace(0, 1, gsy + 1)
gridlines2 = glx = np.linspace(0, 1, gsx + 1)
# Get the centers of each tile in the grid:
gridcentersy = gcy = gly[:-1] + gly[1] / 2
gridcentersx = gcx = glx[:-1] + glx[1] / 2
xg, yg = np.meshgrid(gcx, gcy, indexing='xy')
qs[:] = np.array([xg.ravel(), yg.ravel()]).T
# qsq = qs.reshape((gsy,gsx,2)) # Reshape the coordinates into a 2d array
minor_ticksy = gly
minor_ticksx = glx
major_ticksy = np.arange(0, 1, gly[1] * 5)
major_ticksx = np.arange(0, 1, glx[1] * 5)
# Convert to Improvement function:
gd_i = 1
gd_pts = np.ones((2, 100)) * np.nan
a, b = np.meshgrid((gcx - 0.5), (gcy - 0.5))
local_minima_mask = -a * a - b * b - 1
local_minima_mask *= 0

colors = ['b', 'g', 'y', 'c', 'm', 'r']
t_sim0 = 0
max_cost_idx = 0
vorpoints = np.array(normalized_positions)
Zmeans = np.ones(n_iter) * np.nan
Zstds = np.ones(n_iter) * np.nan


def init_savedir():
    os.mkdir(savedir)
    os.mkdir(os.path.join(savedir, "png"))
    os.mkdir(os.path.join(savedir, "svg"))
    ini_settings = os.path.join(savedir, "coverage.ini")
    with open(ini_filepath, 'r') as file:
        rl = file.readlines()
    with open(ini_settings, 'w+') as file:
        file.writelines(rl)


def load_plot(fig, ax):
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.contour(ans)
    ax.set_xticks(major_ticksx)
    ax.set_xticks(minor_ticksx, minor=True)
    ax.set_yticks(major_ticksy)
    ax.set_yticks(minor_ticksy, minor=True)


def calcF0():
    for j, x in enumerate(gcx):
        dleft = abs(x - 0)
        dright = abs(x - glx[-1])
        dqkx = min(dleft, dright)
        for i, y in enumerate(gcy):
            dlower = abs(y - 0)
            dupper = abs(y - gly[-1])
            dqky = min(dlower, dupper)

            dqk = min(dqkx, dqky)
            if dqk >= c_border:
                bval = - dqk ** 2 / c_border ** 2 + 2 * dqk / c_border
                val = 1 - max(bval, 0)
            else:
                val = 0
            F0[i, j] = val


def recalculate_domain(bounds, mm=False):
    """Recalculates the coordinates of a grid with new domain values..."""
    lx, ly = bounds
    if mm:
        lx = lx / 1000
        ly = ly / 1000
    if glx[-1] != lx or gly[-1] != ly:
        # Scale all the points to the new domain
        if glx[-1] != lx:
            goals_coords[:, :, 0] = goals_coords[:, :, 0] / glx[-1] * lx
            goal_paths[:, :, :, 0] = goal_paths[:, :, :, 0] / glx[-1] * lx
            goal_paths_org[:, :, :, 0] = goal_paths_org[:, :, :, 0] / glx[-1] * lx
        if gly[-1] != ly:
            goals_coords[:, :, 1] = goals_coords[:, :, 1] / gly[-1] * ly
            goal_paths[:, :, :, 1] = goal_paths[:, :, :, 1] / gly[-1] * ly
            goal_paths_org[:, :, :, 1] = goal_paths_org[:, :, :, 1] / gly[-1] * ly
            # goals_coords[:,:,1] = goals_coords[:,:,1]/gly[-1]*ly
        gridlines1[:] = gly[:] = np.linspace(0, ly, gsy + 1)
        gridlines2[:] = glx[:] = np.linspace(0, lx, gsx + 1)
        # Get the centers of each tile in the grid:
        gridcentersy[:] = gcy[:] = gly[:-1] + gly[1] / 2
        gridcentersx[:] = gcx[:] = glx[:-1] + glx[1] / 2

        xg[:], yg[:] = np.meshgrid(gcx, gcy, indexing='xy')
        qs[:] = np.array([xg.ravel(), yg.ravel()]).T
        global major_ticksx, major_ticksy
        major_ticksy = np.arange(0, lx, gly[1] * 5)
        major_ticksx = np.arange(0, ly, glx[1] * 5)

        calcF0()


def update_voronoi_partitioning():
    logger.info("Updating voronoi partitioning")
    if update_voronoi:
        vorpoints[:n_agents, :] = states_all[:n_agents, :2]
    # vor = Voronoi(points)  # update the voronoi partitoning
    kv[:] = whichRegion(qs, vorpoints, kv)
    for m in range(lloyd_max_iterations):
        logger.info(f"lloyd iteration {m}")
        # gx, gy, e, motion = lloyd_step(vorpoints, qs, wheights=1 + np.clip(z_star - Z, 0, z_star))
        gx, gy, e, motion = lloyd_step(vorpoints, qs, wheights=np.clip(M_impr, 0, 1))
        vorpoints[:] = np.array([gx, gy]).T
        logger.debug("Points:")
        [logger.info(f"{_[0]:4.3f},{_[1]:4.3f}") for _ in vorpoints]
        logger.debug(f"energy:{e:6.3f}|motion:{motion:6.3f}|")
    kv[:] = whichRegion(qs, vorpoints, kv)


def sparse_obstacles(agent_idx):
    j = agent_idx
    outside = np.where(k != j, 0, 1)
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((3, 3), np.uint8)
    # n = np.size(np.where(k == j))
    outside = np.array(outside, np.uint8)
    outside2 = cv2.dilate(outside, kernel, iterations=1)
    borderidx = np.where(outside2 - outside != 0)
    left = qsr[:, 0] - qsr[:, 1]
    right = qsr[:, -1] + qsr[:, 1]
    upper = qsr[0, :] - qsr[1, :]
    lower = qsr[-1, :] + qsr[1, :]
    border = qsr[borderidx]
    other_agents = np.delete(states_all[:, :2], j, axis=0)

    other_agents_nn = other_agents[np.where(np.isnan(other_agents[:, 0]) == False)]
    # n2 = np.size(np.where(outside2 != 0))
    allob = np.vstack((left, right, upper, lower, other_agents_nn, border))
    return allob


def init(points, it_num=lloyd_max_iterations, sld_num=5, rho_func=lambda x: 1):
    """Initializes the voroino with a constant gravity field. Based on an implementation of  John Burkardt"""
    n = int(np.mean(sld_num))
    logger.debug(f"Calculating central voronoi points with {n}x{n} granularity")
    whichRegion(qs, points)
    stepi = np.zeros(it_num)
    ei = 1.0E-10 * np.ones(it_num)
    gmi = 1.0E-10 * np.ones(it_num)
    logger.debug(f"Taking {it_num} lloyd iterations")
    if it_num:
        for i in range(it_num):
            gx, gy, e, motion = lloyd_step(points, qs)
            newpoints = np.array([gx, gy]).T
            logger.debug("Points:")
            [logger.debug(f"{_[0]:4.3f},{_[1]:4.3f}") for _ in points]
            logger.debug(f"energy:{e:6.3f}|motion:{motion:6.3f}|")
            logger.debug("Done")

        return newpoints, e, motion
    else:
        return points, None, None


def gradientPath(gradientx, gradienty, field, starting_point, stepsize=path_step_size, extent=[0, 1, 0, 1],
                 epsilon=gradient_epsilon, handrolled=False,
                 agent_pos=None, max_path_points=max_path_points, reverse=True):
    internal_scaler = 100
    p0 = starting_point
    p = p0
    logger.debug(
        f"Gradient from {starting_point[0]:4.2f},{starting_point[1]:4.2f} to {agent_pos[0]:4.2f},{agent_pos[1]:4.2f} in {stepsize:6.3f} steps")
    intpx = lambda x: interpolate.interpn([gcy, gcx], gradientx, [x[1], x[0]], method='nearest', bounds_error=False,
                                          fill_value=None)
    intpy = lambda x: interpolate.interpn([gcy, gcx], gradienty, [x[1], x[0]], method='nearest', bounds_error=False,
                                          fill_value=None)
    if handrolled:
        descent = np.zeros((2, max_path_points)) * np.nan
        descent[:, 0] = starting_point
        dx = stepsize
        dy = stepsize
        delta = stepsize
        for i in range(max_path_points - 1):
            diffsq = (p[0] - agent_pos[0]) ** 2 + (p[1] - agent_pos[1]) ** 2
            if diffsq < epsilon ** 2:
                break
            xx, yy = round_to_grid_idces(p, [gcx, gcy])

            dzx2, dzy2 = gradientx[yy, xx], gradienty[yy, xx]
            # dzx2 = intpx(p)
            # dzy2 = intpy(p)
            grad2 = np.array([dzx2, dzy2]).ravel()
            direction2 = -grad2 / np.linalg.norm(grad2)
            if np.isnan(direction2[0]):
                break

            pnew = p + stepsize * direction2
            descent[:, i + 1] = [np.clip(pnew[0], 0, glx[-1]), np.clip(pnew[1], 0, gly[-1])]
            p = pnew
        # descent[:, i] = agent_pos
        return descent
    else:
        raise NotImplementedError()


def round_to_grid_idces(point, f_gridsize=[gcx, gcy]):
    "Round a point to the closest point in the grid"
    # gridcenters = np.linspace(0,1,f_gridsize+1)[:-1] + 1/f_gridsize/2

    gcx, gcy = f_gridsize
    xdiffs = gcx - point[0]
    ydiffs = gcy - point[1]
    xidx = np.argmin(xdiffs * xdiffs)
    yidx = np.argmin(ydiffs * ydiffs)
    return xidx, yidx


def lloyd_step(points, gridpointsv=qs, k=k, spacing=[gcx, gcy], wheights=None, steps=0):
    gcx, gcy = spacing
    g_num = len(points)
    g = points
    gx = points[:, 0]
    gy = points[:, 1]
    s = gridpointsv

    #  Note that for a nonuniform density, we just set W to the density.
    if wheights is not None:
        wheightsv = wheights.view()

        wheightsv.shape = wheights.size
        w = wheightsv
    else:
        w = np.ones((gcy.size * gcx.size))
    m = np.bincount(kv + 1, w, minlength=n_agents + 1)[1:]  # Count the wheight and discard kv=-1 (int rho dy dx)
    # m = np.bincount(k+1, weights=w)
    #
    #  G is the average of the sample points it is nearest to.
    #  Where M is zero, we shouldn't modify G.
    #
    # sxv,syv =
    gx_new = np.bincount(kv + 1, weights=s[:, 0] * w, minlength=n_agents + 1)[
             1:]  # Count the wheight and discard kv=-1 (int x rho dy dx)
    gy_new = np.bincount(kv + 1, weights=s[:, 1] * w, minlength=n_agents + 1)[
             1:]  # Count the wheight and discard kv=-1 (int y rho dy dx)

    for i in range(0, g_num):
        if (0 < m[i]):
            # p = int p * rho dA / int rho dA
            gx_new[i] = gx_new[i] / float(m[i])
            gy_new[i] = gy_new[i] / float(m[i])
    # #  Compute the energy.
    # #
    e = 0.0
    # for i in range(0, s_num):
    #     e = e + (sx[i] - gx_new[k[i]]) ** 2 \
    #         + (sy[i] - gy_new[k[i]]) ** 2
    #  Compute the generator motion.
    gm = 0
    for i in range(0, g_num):
        gm = gm + (gx_new[i] - gx[i]) ** 2 \
             + (gy_new[i] - gy[i]) ** 2
    return gx_new, gy_new, e, gm


def alpha_unity(p, qr, a_kqi_r, r_icov=cover_radius, spacing=[gcx, gcy]):
    """p: point
        q: generalized coordinates"""
    P = 25
    # r_icov = 0.2 # Radius of coverage
    # Production function:
    p = np.array([p])
    shp = qr.shape
    qv = qr.view()
    # Find indices of points in neighbourhood:
    dx, dy = spacing
    idcx = np.where(abs(dx - p[0, 0]) < r_icov)[0]
    idcy = np.where(abs(dy - p[0, 1]) < r_icov)[0]
    qv.shape = (shp[0] * shp[1], shp[2])
    qboundedrect = qr[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1, :].copy()
    qboundedrectv = qboundedrect.view()
    qboundedrectv.shape = (qboundedrect.shape[0] * qboundedrect.shape[1], qboundedrect.shape[2])
    qboundedrectv -= p
    r = np.linalg.norm(qboundedrect, axis=2)
    # 1 inside, 0 everywhere else:
    covalue = np.where(r <= r_icov, 1, 0)
    a_kqi_r[:] = 0
    a_kqi_r[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1] = covalue
    return a_kqi_r


def alpha_sq(p, qr, a_kqi_r, r_icov=cover_radius, spacing=[gcx, gcy]):
    """p: point
        q: generalized coordinates"""
    P = 25
    # r_icov = 0.2 # Radius of coverage
    # Production function:
    p = np.array([p])
    shp = qr.shape
    qv = qr.view()
    # Find indices of points in neighbourhood:
    dx, dy = spacing
    idcx = np.where(abs(dx - p[0, 0]) < r_icov)[0]
    idcy = np.where(abs(dy - p[0, 1]) < r_icov)[0]
    qv.shape = (shp[0] * shp[1], shp[2])
    qboundedrect = qr[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1, :].copy()
    qboundedrectv = qboundedrect.view()
    qboundedrectv.shape = (qboundedrect.shape[0] * qboundedrect.shape[1], qboundedrect.shape[2])
    qboundedrectv -= p
    r = np.linalg.norm(qboundedrect, axis=2)

    # covalue = P * (r_icov**2 - (np.clip(r,0,r_icov)) ** 2)/r_icov**2
    covalue = P / r_icov * (np.clip(r, 0, r_icov) - r_icov) ** 2

    a_kqi_r[:] = 0
    a_kqi_r[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1] = covalue

    return a_kqi_r


def alpha_i(p, qr, a_kqi_r, r_icov=cover_radius, spacing=[gcx, gcy]):
    """p: point
        q: generalized coordinates"""
    P = 25
    # r_icov = 0.2 # Radius of coverage
    # Production function:
    p = np.array([p])
    shp = qr.shape
    qv = qr.view()
    # Find indices of points in neighbourhood:
    dx, dy = spacing
    idcx = np.where(abs(dx - p[0, 0]) < r_icov)[0]
    idcy = np.where(abs(dy - p[0, 1]) < r_icov)[0]
    qv.shape = (shp[0] * shp[1], shp[2])
    qboundedrect = qr[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1, :].copy()
    qboundedrectv = qboundedrect.view()
    qboundedrectv.shape = (qboundedrect.shape[0] * qboundedrect.shape[1], qboundedrect.shape[2])
    qboundedrectv -= p
    r = np.linalg.norm(qboundedrect, axis=2)

    covalue = P * (r_icov ** 2 - (np.clip(r, 0, r_icov)) ** 2) / r_icov ** 2
    # covalue = P / r_icov * (np.clip(distances, 0, r_icov) - r_icov) ** 2

    a_kqi_r[:] = 0
    a_kqi_r[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1] = covalue

    return a_kqi_r


def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define a connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # neighborhood = np.ones((5,5),np.bool)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = np.logical_xor(local_min, eroded_background)
    return np.where(detected_minima)


#
# def update_goals(update_list):
#
#     for l in update_list:  # Mask the values in the current partion:
#         zerosq = np.zeros(grid_size ** 2)
#         M_impr_mask = np.where(k == l, M_impr.reshape(gsx * gsy), 0)
#         Msq = -M_impr_mask.reshape(gsx * gsy)
#         max_i, max_j = detect_local_minima(Msq)
#         for idx, pair in enumerate(zip(max_i, max_j)):
#             goals[l, idx, :] = pair
#             coord = [gcx[pair[0]],gcy[pair[1]]]
#             goals_coords[l, idx, :] = coord
# #
# def calc_Mi_fast(p,qr,):
#     Alpha_i[:] = alpha_i(p, qr, Alpha_i, r_icov=cover_radius, spacing=[gcx, gcy]) #2d grid with prod fcn
#     Bi = np.sum(phi*rho_max*Alpha_i)
#     Ai = Bi**-1 * np.sum(decay*phi*rho_max*Alpha_i)
#
#     Mi_new = (1-decay)*M_impr + Ai
#
#     p = np.array([p])
#     shp = qr.shape
#     qv = qr.view()
#     # Find indices of points in neighbourhood:
#     dx, dy = spacing
#     idcx = np.where(abs(dx - p[0, 0]) < r_icov)[0]
#     idcy = np.where(abs(dy - p[0, 1]) < r_icov)[0]
#     qv.shape = (shp[0] * shp[1], shp[2])
#     qboundedrect = qr[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1, :].copy()
#     qboundedrectv = qboundedrect.view()
#     qboundedrectv.shape = (qboundedrect.shape[0] * qboundedrect.shape[1], qboundedrect.shape[2])
#     qboundedrectv -= p
#     r = np.linalg.norm(qboundedrect, axis=2)
#
#
#     covalue = P * (r_icov**2 - (np.clip(r,0,r_icov)) ** 2)/r_icov**2
#     # covalue = P / r_icov * (np.clip(distances, 0, r_icov) - r_icov) ** 2
#
#     a_kqi_r[:] = 0
#     a_kqi_r[idcy[0]:idcy[-1] + 1, idcx[0]:idcx[-1] + 1] = covalue
#
#     return a_kqi_r
#
#


def calc_Mi(alpha_i=alpha_i):
    """Calculates the grid representing the improvement function, note that this is a naive approach, and could be sped up
    by using the iterative update algo using dM"""
    Zi = Z
    Znorm = -Zi / z_star + 1
    # Znorm = Znorm.ravel()
    M_impr[:] = M_impr * 0
    shp = qsr.shape
    Mpv = M_impr.view()
    Mpv.shape = (shp[0] * shp[1])
    qv = qsr.view()
    qv.shape = (shp[0] * shp[1], shp[2])
    a_kqi = np.zeros_like(Zi)
    for idx, p in enumerate(qv):
        a_kqi[:] = alpha_i(p, qsr, a_kqi)  # a on point p when an agent is located at q
        num = np.multiply(Znorm, a_kqi)
        den = a_kqi
        Mpi = np.sum(num) / np.sum(den)
        Mpv[idx] = Mpi
        # # Alternative calculation, could be faster:
        # nonzero = np.where(a_kqi!=0)
        # num = np.multiply(Znorm[nonzero],a_kqi[nonzero])
        # den = a_kqi[nonzero]
        # Mpi = np.sum(num)/np.sum(den)
    return M_impr


def calculate_path_score(M_impr, path, spacing=[gcx, gcy], length_cost=length_mp, max_len=max_spline_points, j=0):
    """Calculates L_i cost of the path, given a certain field"""
    cost = 0
    M_2d = M_impr
    gcx, gcy = spacing
    intpfn = lambda x: interpolate.interpn([gcy, gcx], M_2d, [x[1], x[0]], method='nearest', bounds_error=False,
                                           fill_value=None)

    max_goals = np.delete(goals_coords_max, j, axis=0)
    nn = np.where(np.isnan(max_goals[:, 0]) == False)
    ppoints = max_goals[nn, :]
    max_goals = ppoints

    if max_goals.size and path.size:
        end_point = path[:, 0].T
        dists = np.linalg.norm(max_goals - end_point, axis=2)
        if np.min(dists) < goal_radius_prox:  # Safe radius with some spacing
            return 0
    for p in path.T:
        try:
            improvement = intpfn(p)
            if improvement > 1:
                logger.debug("")
            # cost += max(0, improvement)
            cost += improvement
        except IndexError as e:
            logger.error(e)
    # ltot = np.diff(path[:,0])**2 + np.diff(path[:,1])**2

    l = np.size(path) // 2
    if l > 1:
        try:
            diffx, diffy = np.diff(path[0, :]), np.diff(path[1, :])
            lt = np.sum(diffx ** 2 + diffy ** 2) ** .5  # total path length

            final_cost = cost / l + length_cost * lt
            return float(final_cost)
        except Exception as e:
            logger.error(e)
    elif l == 1:
        return float(cost)
    else:
        return 0.0


def calc_rho(ai):
    """Calculates the value of rho_star given the field Z, the wheighing phi and production function ai"""
    zdelta = z_star - (1 - decay) * Z
    phi = 1
    num = np.sum(phi * zdelta * ai)
    den = np.sum(phi * ai ** 2)

    rho_star = num / den
    # logger.info(rho_star)
    return np.clip(rho_star, 0, rho_max)


def add_coverage(Z, x, dt, km=rho_max, qsr=qsr):
    """Adds coverage to the Z-field"""
    # alpha_i_vec = alpha_i(x[:2],qs,r_icov=cover_radius)
    if x[0] > 0 and x[1] > 0 and x[0] < glx[-1] and x[1] < gly[-1]:
        Alpha_i[:] = alpha_i(x, qsr, Alpha_i)
        if dynamic_rho:
            rho_star = calc_rho(Alpha_i)
            Z += Alpha_i * rho_star * dt
        else:
            Z += Alpha_i * rho_max * dt
    else:

        logger.debug("Agent out of bounds")
    return Z


def update_partition_and_goal(i_reached):
    updatelist = i_reached
    if not len(updatelist):
        # We must recalculate the goals so there is an update from M_impr needed
        return True, []
    tc = time.perf_counter()
    M_impr[:] = calc_Mi()
    logger.info(f"updating M_Impr took {time.perf_counter() - tc:6.3f}s")
    update_voronoi_partitioning()
    for l in range(n_agents):  # Mask the values in the current partion:

        if l in updatelist:
            continue
        else:
            # pend2 = goal_paths_selected[l, :, 0]
            pendnn = goal_paths[l, goals_max_index[l], goals_index[l]:]
            idc = np.where(np.isnan(pendnn) == True)
            if (np.isnan(pendnn) == True).any():
                updatelist.extend([l])
            elif not pendnn.size:
                updatelist.extend([l])
            else:
                if len(pendnn.shape) == 1:
                    reg = np.array([1], np.int16) * 0 - 1
                    reg[:] = whichRegion([pendnn], vorpoints, reg)
                else:
                    reg = np.ones_like(pendnn[:, 0], np.int16) * -1
                    reg[:] = whichRegion(pendnn, vorpoints, reg)
                invalids = np.where(reg != l)
                iv = np.array(invalids).ravel()
                if not np.size(iv):
                    logger.info(f"goal in region {l} still valid")
                    # logger.debug("Goal is still in valid region")
                elif iv[0] > 0:
                    # Some points of the goal are invalid. Delete the tip of the path.f
                    logger.debug("Some points are stil valid!")
                    # firtInvalid = iv[0]
                    # shift = int(pendnn.size//2 - iv[0])
                    # fill = np.full((shift,2), np.nan)
                    # logger.debug(f"Snipping {shift} of the tip of the goal path of {l}: {firtInvalid}")
                    # sharr = goal_paths[l, max_cost_idx,:-shift]
                    #
                    # narr = np.vstack((fill,sharr))
                    # goal_paths[l, max_cost_idx,:]= narr
                    # goals_index[l]=goals_index[l]+shift
                else:
                    logger.debug(f"We need recalculation for {l}")
                    updatelist.extend([l])
    # Updating updatelist
    logger.info(f"updating {updatelist}")

    for l in updatelist:  # Mask the values in the current partion:
        # if i_reached == l:
        # reset goals
        goals_index[l] = 0  # reset this path index
        goals_scores[l, :] = np.nan  # reset this path values
        goals[l, :] = goals[l, :] * np.nan
        goal_paths[l, :] = goal_paths[l, :] * np.nan
        goal_paths_org[l, :] = goal_paths_org[l, :] * np.nan
        FMMgrids[l, :] = FMMgrids[l, :] * np.nan
        # FMMcombined[:] = FMMcombined * np.nan

        M_impr_mask = np.where(k == l, M_impr, 0)
        M_impr_mask2 = np.where(M_impr_mask > 0, M_impr, 0)
        # Msq[0:12,7:12] = Msq[0:12,7:12]*0
        # M_a =
        # smoothed_array = sp.ndimage.filters.gaussian_filter(-M_impr_mask, [1,1], mode='constant')
        # M_impr_mask = np.where(k == l, smoothed_array, 0)
        # smoothed_array_2 = np.where((k==l),smoothed_array,0)
        # np.logical_xor(smoothed_array)
        max_y_i, max_x_i = detect_local_minima(-M_impr_mask2)  # Minus since we are looking for maxima in the improv fcn
        if not max_y_i.size:
            # Relax constraints, also search in overcovered area's to at least find a goal
            smoothed_array = sp.ndimage.filters.gaussian_filter(-M_impr_mask, [3, 3], mode='constant')
            M_impr_mask = np.where(k == l, smoothed_array, 0)
            # M_impr_mask = np.where(k == l, M_impr, 0)
            max_y_i, max_x_i = detect_local_minima(-M_impr_mask)
            logger.info(f"Everything in region {l} is well covered!")
        for idx, pair in enumerate(zip(max_x_i, max_y_i)):
            xc = gcx[max_x_i]
            yc = gcy[max_y_i]
            val = -M_impr[pair[1], pair[0]]
            goals[l, idx % max_num_goals, :] = pair
    FMM_speedfcn = (1 + np.exp(-beta * M_impr)) ** -1
    Ftot[:] = FMM_speedfcn + F0

    for l in updatelist:  # Calculate the travel times towards each goal for each agent position:
        agent_pos = states_all[l, :2]
        start_idcesxy = round_to_grid_idces(agent_pos)  # Find the index of the grid tile closest to the agent position

        # Testregion is the region the test point is in:
        # region_idx = k[start_idcesxy[1], start_idcesxy[0]]
        region_idx = l
        mask = np.where(k == region_idx, True, False)
        # Define the contour for the FMM algo
        phi = np.ones((gsy, gsx), np.int32)
        # phi[eids[0],eids[1]] = -1 #This point must lie within the test region
        cy, cx = [start_idcesxy[1], start_idcesxy[0]]
        phi[cy, cx] = -1  # This point must is the origin of the wave == agent position
        speed = Ftot.copy()
        speed = np.where(k == region_idx, Ftot, 0.01)
        for i in range(-1, 2):
            for j in range(-1, 2):
                x, y = np.clip(cx + i, 0, gsx - 1), np.clip(cy + j, 0, gsy - 1)
                speed[y, x] = np.max((0.1, speed[
                    y, x]))  # To avoid negative time marching around contour, avoid singular speed values #issue18

        phi_mask = np.ma.MaskedArray(phi, ~mask)  # limit the calculations only to this region
        valids = phi_mask[~phi_mask.mask]
        phi_mask = phi  # this line disables the masking.
        # phi_mask = np.where(k != region_idx, True, False)  # this line disables the masking.
        dc = [gcy[1] - gcy[0], gcx[1] - gcx[0]]
        try:
            if len(valids) > 0:
                if np.min(valids) == -1:
                    # fmm_times = skfmm.travel_time(phi_mask, speed, dx=dc, order=1)
                    fmm_times = skfmm.travel_time(phi, speed, dx=dc, order=1)  # this line disables the masking.
                    ygrad, xgrad = np.gradient(fmm_times)
                    FMMgrids[region_idx] = fmm_times
                    FMMgradients[region_idx, :, :, 0] = xgrad
                    FMMgradients[region_idx, :, :, 1] = ygrad
                    idces = np.where(k == region_idx)
                    newtimes = fmm_times[idces]
                    if newtimes.size:
                        max_fmm = np.max(fmm_times[idces])
                        FMMcombined[idces] = newtimes  # Pretty combined FMM for plotting
        except Exception as e:
            logger.error(e)
        if region_idx != l:
            logger.error("whut")

        # FMMcombined[idces] = fmm_times*(1-mask)*(1-mask)
    return True, updatelist


def sanitize_path(path):
    nanidx = np.where(np.isnan(path[0, :]) == False)[0]  # Delete NaN's
    pathnn = path[:, nanidx]

    for i, pathpoint in enumerate(path.T):
        reg = np.array([1]) * 0 - 1
        # Clip to domain:
        p0 = np.clip(pathpoint[0], glx[0], glx[-1])
        p1 = np.clip(pathpoint[1], gly[0], gly[-1])
        pclip = np.array([p0, p1])
        path[:, i] = pclip


def calc_paths(update_list):
    # Calculates goal scores.
    # M_impr_delta
    for j in update_list:  # iterate over agents to calculate goal paths
        goals_scores[j] = np.ones((max_num_goals)) * np.nan
        # goals_index     = np.zeros((n_agents,max_num_goals),np.uint16)
        goal_paths[j] = np.ones((max_num_goals, max_spline_points, 2)) * np.nan
        goal_paths_org[j] = np.ones((max_num_goals, max_path_points, 2)) * np.nan
        goals_index[j] = 0  # Start covering at the beginning of the path
        goals_max_index[j] = 0
        max_goal_score = -10e3
        logger.debug(f"New paths for {j}")
        for g in range(max_num_goals):  # Iterate over goals
            goal = goals[j, g]
            if not np.isnan(goal[0]):
                goal_coord = [gcx[int(goal[0])], gcy[int(goal[1])]]
                goals_coords[j, g, :] = goal_coord
                try:
                    agent_pos = states_all[j, :2]
                    start_idcesxy = round_to_grid_idces(agent_pos)
                    ongrid = [gcy[start_idcesxy[1]],
                              gcx[start_idcesxy[0]]]  # Rounded position of the agent to the nearest grid point
                    descent = gradientPath(FMMgradients[j, :, :, 0], FMMgradients[j, :, :, 1], FMMgrids[j],
                                           goal_coord,
                                           stepsize=path_step_size,
                                           epsilon=gradient_epsilon,
                                           handrolled=True,
                                           agent_pos=agent_pos,
                                           max_path_points=max_path_points,
                                           reverse=False)

                    nanidx = np.where(np.isnan(descent[0, :]) == False)[0]
                    path = descent[:, nanidx]
                    # goal_paths_org[j, g, :, :] = path.T[::-1]
                    if path.size:
                        goal_paths_org[j, g, - path.size // 2:, :] = path.T[::-1]

                    for i, pathpoint in enumerate(path.T):
                        reg = np.array([1]) * 0 - 1
                        p0 = np.clip(pathpoint[0], gcx[0], gcx[-1])
                        p1 = np.clip(pathpoint[1], gcy[0], gcy[-1])
                        pclip = np.array([p0, p1])
                        path[:, i] = pclip
                        # whichRegion([pathpoint], vorpoints, reg)
                        # if reg[0] != j:
                        #     inside = qs[np.where(kv == j)]
                        #     cpi = dwa.closestPoint(inside, pathpoint)
                        #     cp = inside[cpi]
                        #     path[:, i] = cp

                    spath, idc = np.unique(path[0, :], return_index=True)  # Shorten path to remove dupes
                    idcsu = np.sort(idc)

                    def delete_closepoints(path2d, epsilon=1e-10):
                        if path2d.size <= 2:
                            return True, path2d
                        try:
                            idcs = np.where(np.diff(path2d[0, :]) ** 2 + np.diff(path2d[1, :]) ** 2 < epsilon)
                        except IndexError as e:
                            logger.error(e)
                            # return True, path2d
                        lc = len(idcs[0])
                        if lc:
                            npicds = np.where(np.diff(path2d[0, :]) ** 2 + np.diff(path2d[1, :]) ** 2 >= epsilon)
                            npicds_new = np.hstack((npicds[0], np.array(len(path2d.T) - 1)))
                            newpath2d = path[:, npicds_new].squeeze()
                            return delete_closepoints(newpath2d, epsilon)
                        else:
                            return True, path2d

                    setpath = path
                    rv, setpath = delete_closepoints(path)
                    lp = np.size(setpath) // 2
                    # idcs = np.where(np.diff(path[0,idcsu])**2>1e-10) #Remove points very close to eachother
                    # setpath = path[:,idcs].squeeze()
                    # goal_paths[j,g,max_path_points-len(idc):,:] = setpath.T[::-1]

                    try:

                        n_pts = int(np.clip(float(lp) / max_path_points * max_spline_points, 2,
                                            max_spline_points))  # Cap number of points
                        # n_pts = int(lp)
                        # try:
                        #     diffx, diffy = np.diff(setpath[0, :]), np.diff(setpath[1, :])
                        # except IndexError as e:
                        #     logger.warning(e)
                        # lengths = (diffx ** 2 + diffy ** 2) ** .5  # total path length

                        # Simplify path as a spline:
                        # lp = -1 #disable interpolation
                        if lp > 2:  # minimum of 3 points needed for a spline
                            if lp == 3:
                                tck, u = interp.splprep(setpath, s=0, k=1)  # Find a spline going through all points
                            else:
                                tck, u = interp.splprep(setpath, s=0, k=3)  # Find a spline going through all points
                            # n_pts = max(2,min(len(idc)//5,max_path_points))# Cap number of points
                            u2 = np.linspace(0.0, 1.0, n_pts)
                            # u2 =u
                            spline_path = np.column_stack(interp.splev(u2, tck)).T
                            goal_paths[j, g, max_spline_points - n_pts:, :] = spline_path.T[::-1]
                        elif lp == 1:
                            goal_paths[j, g, -1, :] = pathpoint
                            spline_path = np.array([pathpoint]).T
                        elif lp == 0:
                            spline_path = np.array([])
                        else:
                            spline_path = setpath
                            goal_paths[j, g, max_spline_points - 2:, :] = spline_path.T[::-1]

                    except Exception as e:
                        logger.error(e)
                        raise e

                    score = calculate_path_score(M_impr, spline_path, [gcx, gcy], j=j, length_cost=length_mp,
                                                 max_len=max_spline_points)
                    goals_scores[j, g] = score
                    if score > max_goal_score:
                        logger.debug(f"Goal {g} has a score of {score:4.2f}>{max_goal_score:4.2f}")
                        max_goal_score = score
                        goals_max_index[j] = g
                        goals_coords_max[j] = goal_coord
                except SystemError as e:
                    logger.error(e)
    return goals_scores, goal_paths


def next_path(x, j, goals, config, idx):
    """Calculates the closest goal path point that is not yet reached"""
    rv = False
    if idx == max_spline_points:
        """All intermediate points are reached"""
        return True, 0
        logger.warning("Out of bounds")
    target = goals[:, idx]
    dist_to_goal = np.linalg.norm(x[:2] - target)

    if np.isnan(dist_to_goal):
        """NaN point, move to next point"""
        idx += 1
        rv, idx = next_path(x, j, goals, config, idx)
    elif dist_to_goal >= config.goal_point_radius:
        """Next point is not yet reached, check if its still in our region"""
        l = goals[:, idx:].size // 2
        reg = np.array([1], np.int16) * 0 - 1
        kt = whichRegion([target], vorpoints, reg)[0]  # Region targer is  in
        kc = whichRegion([x[:2]], vorpoints, reg)[0]  # Region we are in
        kp = j  # Region we should be in

        if kp != kt:
            return True, idx
    else:
        idx += 1
        rv, idx = next_path(x, j, goals, config, idx)
    # if dist_to_goal <= config.goal_point_radius:
    #     l =goals[:, idx:].size//2
    #     reg = np.array([1], np.int16) * 0 - 1
    #     kt = whichRegion([target], vorpoints, reg)[0]
    #     kp = whichRegion([x[:2]], vorpoints, reg)[0]
    #     if l > 1:
    #         # logger.debug(f"{l}")
    #         logger.debug(f"{l}/{len(goals.T)} Goal!!")
    #         idx += 1
    #         rv, idx = next_path(x, goals, config, idx)
    #     elif kp != kt and not np.isnan(target[0]):
    #         rv = True
    #     else:
    #         # We have exhausted this path, delete the goal and move to next....
    #         rv = True
    return rv, idx


def plot_fcn(fig, ax, vorpoints, t_sim0, Z=Z, M_impr=M_impr, i_iter=0, savename="out", save=True):
    ax.clear()
    gnni = np.where(np.isnan(vorpoints[:n_agents, 0]) == False)
    plotvorpoints = vorpoints[gnni]
    vor = Voronoi(plotvorpoints)  # update the voronoi partitoning
    indomains = np.array(gnni[0], np.uint8)
    logger.info(f"Plotting {savename} | {t_sim0:4.2f}")
    if show_dw:
        for j in indomains:  # iterate over agents
            x = states_all[j]
            dwa.plot_agent(x, dwa.config, ax)
            dwa.plot_dw(x, dwa.config, ax)
            dwa.plot_arrow(x[0], x[1], x[2])
            traj = states_all_traj[j, :]
            endp = np.max(np.where(np.isnan(traj[:, 0]) == False))
            begp = max(0, endp - max_plot_trajectory_points)
            ax.plot(traj[begp:endp, 0], traj[begp:endp, 1], "-", color=colors[j])
    if show_optimal_paths:
        for j in indomains:  # iterate over agents
            # max_cost = np.max(goals_scores[j, np.where(np.isnan(goals_scores[j, :]) == False)])
            max_idx = goals_max_index[j]
            for g in range(max_num_goals):  # Iterate over goals
                path = goal_paths[j, g, :, :]
                path_org = goal_paths_org[j, g, :, :]
                nanidx = np.where(np.isnan(path[:, 0]) == False)[0]
                # path_nonan = path[nanidx, :]
                score = goals_scores[j, g]
                strcost = f"Goal:{g}  {nanidx.size}/{max_spline_points}--{float(goals_scores[j,g]):4.2f}"
                if nanidx.size:  # The goal exists:

                    pend = path_org[-1, :]
                    if max_idx == g:
                        ax.plot(*path_org.T, '-.', color='r')
                        ax.plot(*path_org.T, 'x', color='r')
                        ax.plot(*path.T, '+', color='k')
                        next_goal = goal_paths[j, g, goals_index[j], :].T
                        ax.scatter(*next_goal, 50, color='g')
                        if show_goal_text:
                            ax.text(*pend, strcost, color='r')
                    else:
                        ax.plot(*path_org.T, '-.', color=colors[j], alpha=0.5)
                        if show_goal_text:
                            ax.text(*pend, strcost)
    if show_velo_field:
        ax.imshow(Ftot, extent=[0, glx[0], 0, gly[0]], alpha=0.6, origin='lower')
    if show_fmm_grids:
        ax.imshow(FMMcombined, extent=[0, 1, 0, 1], alpha=0.6, origin='lower', vmin=0, vmax=0.35)
        # ax.imshow(FMMgrids[0], extent=[0, 1, 0, 1], alpha=0.6, origin='lower')
        # ax.imshow(FMMgrids[2], extent=[0, 1, 0, 1], alpha=0.6, origin='lower')
        # ax.contour(FMMcombined)
    if show_coverage_field:
        cov = ax.imshow(Z, extent=[0, glx[0], 0, gly[0]], alpha=0.6, origin='lower', vmin=0, vmax=2 * z_star)
        # ax.contour(gcx,gcy,Z)
    if show_improvement:
        ax.imshow(1 - M_impr, extent=[0, glx[0], 0, gly[0]], alpha=0.6, origin='lower', vmin=0, vmax=1)
        # ax.contour(1-M_impr.reshape((grid_size,grid_size)))
        # ax.imshow(Z,extent=[0,1,0,1],alpha=0.6,origin='lower')
    if show_grid_points:
        for i in range(n_agents):
            idces = np.where(kv == i)
            ck = qs[idces]
            ax.scatter(*ck.T, int(200 * gcx[0]), alpha=0.5, color=colors[i])
    if show_grid_lines:
        # And a corresponding grid
        ax.grid(which='both')
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=1)
    # ax.imshow(alphas.reshape(grid_size,grid_size).astype(np.uint8).T,extent=[0,1,0,1],alpha=0.2,origin='lower')
    if show_init_points:
        npos = np.array(normalized_positions)
        ax.scatter(npos[:, 0], npos[:, 1], 100, "k", marker='d')
    voronoi_plot_2d(vor, ax)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    Zmean = np.sum(Z) / Z.size
    Zstd = np.std(Z)

    ax.set_title(f"t = {t_sim0:4.2f}: Z_mean= {Zmean:4.2f}/100, sigma = {Zstd:4.2f} ")

    # plt.savefig()
    if save:
        fig.canvas.draw()
        savepathpng = os.path.join(savedir, "png", f"{savename}.png")
        savepathsvg = os.path.join(savedir, "svg", f"{savename}.svg")

        fig.savefig(fname=savepathsvg)
        fig.savefig(fname=savepathpng)
        image = cv2.imread(savepathpng)
        cv2.imshow("Feed", cv2.resize(image, (800, 800)))
        cv2.waitKey(1)


def whichRegion(qs, cvt_points, kv=kv):
    """Search for the region were the points are situated"""

    for i, x in enumerate(cvt_points):
        sc = 3 * (i + 1)
        if x[0] > glx[0] and x[1] > gly[0] and x[0] < glx[-1] and x[1] < gly[-1]:
            logger.debug("We are in the domain")
        else:
            cvt_points[i] = [np.clip(cvt_points[i, 0], glx[0], glx[-1]), np.clip(cvt_points[i, 1], gly[0], gly[-1])]
    gnni = np.where(np.isnan(cvt_points[:, 0]) == False)
    s, g = qs, cvt_points[gnni]
    if cvt_points[gnni].size == 2:
        kv[:] = kv * 0 + gnni[0][0]
        return kv
    elif cvt_points[gnni].size < 2:
        kv[:] = kv * 0 - 1
        return kv
    kv[:] = kv * 0 - 1
    for idx, ss in enumerate(s):
        if np.isnan(ss[0]):
            kv[idx] = -1
            continue
        if ss[0] > 1 or ss[1] > 1 or ss[0] < 0 or ss[1] < 0:
            kv[idx] = -1
        minvals = [np.inner(gg - ss, gg - ss) for gg in g]
        mindx = np.argmin(minvals)
        minl = gnni[0][mindx]
        kv[idx] = minl
    return kv


def initCoverageControl():
    # Initialize the coverage control algorithm
    points = np.array(normalized_positions)  # Convert point to an array
    states_all[:n_agents, :2] = points[:n_agents, :]
    # Do a voronoi iteration
    newvorpoints, energy, motion = init(points, lloyd_max_iterations,
                                        grid_size)  # Do a lloyd iteration if needed to move towards a cvt.
    vorpoints[:] = newvorpoints
    for i in points:  # Init the field with a very low field to reduce the local minima points that are all 0
        if not np.isnan(i[0]):
            Alpha_i[:] = alpha_i(i, qsr, Alpha_i, r_icov=1)
            Zstart[:] = Zstart + Alpha_i
    # Reshape the grid and normalize
    Zstart_max = np.max(Zstart)
    Z0 = Zstart / np.max(Zstart) * z_star / 100  # 1 % of z_star
    Z[:] = Z0.copy()
    calcF0()


def monitor_distances(agent_idx):
    j = agent_idx
    idx = int(goals_index[j])
    max_cost_idx = goals_max_index[j]
    dwa_goals = goal_paths[j, max_cost_idx, idx:].T
    # j = agent_idx
    endpoint = dwa_goals[:, -1]
    endpoint2 = np.array(goals_coords[j, max_cost_idx])

    errgoal = np.linalg.norm(endpoint - endpoint2)
    begin = dwa_goals[:, 0]
    xcur = states_all[j, :2]
    errbegin = np.linalg.norm(begin - xcur)
    # if errgoal > dwa.config.robot_radius*2 or errbegin > dwa.config.robot_radius*max_path_points/max_spline_points*2:
    if errgoal > 0.1 or errbegin > 0.1:
        logger.warning(f"{j}Path is too far away from agent position! {errgoal:4.2f} and {errbegin:4.2f}")
        logger.warning(f"{j}Path is too far away from agent position! {errgoal:4.2f} and {errbegin:4.2f}")
        # logger.debug("")


def control_step(i_iter, dt=dwa.config.dt):
    update_list = []
    if np.where(np.isnan(vorpoints[:, 0]))[0].size == vorpoints.size // 2:
        update_voronoi_partitioning()
        logger.info("Nothing inside domain.. updating voronoi.....")

    for j in range(n_agents):
        xvor = vorpoints[j]
        if np.isnan(xvor[0]):
            continue  # We are outside the domain, disable the goal searching
        x = states_all[j]
        if states_all[j, 0] == 0 and states_all[j, 1] == 0 and states_all[j, 0] == 0:
            continue  # We are not initialized
        logger.debug(f"Calculating agent paths for {j} on {x[0]:4.2f},{x[1]:4.2f}")
        # nonans = np.where(np.isnan(goals_scores[j, :]) == False)
        max_cost_idx = goals_max_index[j]
        idx = int(goals_index[j])
        current_goal = goal_paths[j, max_cost_idx].squeeze().T
        pathfollowed, idx = next_path(x, j, current_goal, dwa.config, idx)
        goals_index[j] = idx  # Update the index to be travelled upwards
        if pathfollowed:
            update_list.append(j)
            logger.info(f"Agent {j} reached goal, others are are at: {goals_index}/{max_spline_points-1}")
            continue

        logger.debug(
            f"Moving from {x[0]:4.2f},{x[1]:4.2f} to {current_goal[:, idx]} to goal {max_cost_idx} at {current_goal[:, -1]}")
        try:
            monitor_distances(j)
            ob = sparse_obstacles(j)
            # ob = None
            u, predicted_trajectory = dwa.dwa_control(x, dwa.config, current_goal[:, idx:], ob=ob)
            states_all[j, 3:5] = u
        except IndexError as e:
            logger.error(e)
            raise e
        # states_all[j] = x

    return update_list


def main():
    if screenshots or end_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        if show_coverage_field:
            cov = ax.imshow(Z, extent=[0, 1, 0, 1], alpha=0.6, origin='lower', vmin=0, vmax=2 * z_star)
            bar = fig.colorbar(cov, shrink=0.8, aspect=10)
        elif show_fmm_grids:
            grid = ax.imshow(FMMcombined, extent=[0, 1, 0, 1], alpha=0.6, origin='lower')
            bar = fig.colorbar(grid, shrink=0.8, aspect=10)
        load_plot(fig, ax)

        # logger = logging.getLogger("planebots")

    logger = logging.getLogger(__name__)
    logger.addHandler(log_short)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # Get the initial agent positions.
    logger.info("Starting program...")
    tic = time.perf_counter()
    t_sim0 = 0
    logger.info("Beginning loop")
    initCoverageControl()

    M_impr[:] = calc_Mi()
    # M_impr[0:5,3:5] = M_impr[0:5,3:5]*0-1  # add covered points <0 is overcovered
    # M_impr[0:5,5:7] = np.clip(M_impr[0:5,5:7]*0+1,0,1)# add undercovered points
    kv[:] = whichRegion(qs, states_all[:, :2], kv)
    i_reached = [0]
    rv, update_list = update_partition_and_goal(i_reached)
    goals_scores, goal_paths = calc_paths(update_list)
    EXIT_FLAG = False
    exit_iter = 0
    for i in range(n_iter - 1):
        t_sim0 += dwa.config.dt
        logger.info(f"simulation of {t_sim0:4.2f} run in {time.perf_counter() - tic:4.2f}")
        if EXIT_FLAG:
            break
        Zmeans[i] = np.sum(Z) / Z.size
        Zstds[i] = np.std(Z)
        Z[:] = Z * (1 - decay * dwa.config.dt)
        update_list = control_step(i, dwa.config.dt)
        if len(update_list):
            exit_iter += 1  # An invalid goal detected, switching goals
            logger.info(f"Iteration {exit_iter}/{max_goal_switches}, max cost goal:{max_cost_idx}")
            # update_voronoi()
            rv, update_list = update_partition_and_goal(update_list)
            calc_paths(update_list)
        for j in range(n_agents):
            states_all_traj[j, i] = states_all[j]
            x = states_all[j]

            xn = dwa.motion(x, [x[3], x[4]], dwa.config.dt)  # simulate robot
            states_all[j] = xn
            Z[:] = add_coverage(Z, x[:2], dwa.config.dt)
        if screenshots:
            if screenshots_from <= t_sim0:
                plot_fcn(fig, ax, vorpoints, t_sim0=t_sim0, Z=Z, M_impr=M_impr, i_iter=i, save=True,
                         savename=f"out{i:04d}")
        if exit_iter == max_goal_switches - 1:
            logger.info("Last iteration")
        if exit_iter == max_goal_switches:
            EXIT_FLAG = True

    logger.info(f"simulation of {t_sim0} run in {time.perf_counter() - tic}")
    if not screenshots and end_plot:
        # Update all for pretty pics of FMM
        rv, update_list = update_partition_and_goal(np.arange(0, n_agents, 1))
        calc_paths(update_list)
        plot_fcn(fig, ax, vorpoints, t_sim0=t_sim0, Z=Z, M_impr=M_impr, i_iter=i, save=True)
        # plot_fcn(ax, vorpoints, t_sim0=t_sim0, Z=Z, M_impr=M_impr, i_iter=i)
        plt.show()

    fig2, ax = plt.subplots()
    ax.plot(Zmeans)
    ax.plot(Zstds)
    ax.set_ylim([0, 2 * z_star])
    fig.savefig(fname=os.path.join(savedir, "Field.pdf"), dpi=200)
    logger.debug("Done")
    plt.show()


if __name__ == '__main__':
    init_savedir()
    recalculate_domain(domain_mm, True)
    main()

    saveArrays = {
        'states_all_traj': states_all_traj,
        'Zmeans': Zmeans,
        'Zstds': Zstds}
    np.savez_compressed(savename, **saveArrays)
