import numpy as np
import scipy
from scipy.spatial.distance import euclidean


def distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return euclidean(v1.flatten(), v2.flatten())


def magnitude(v):
    return np.maximum(1e-12, np.sqrt(v.dot(v)))


def normalize(v):
    return v / magnitude(v)


def calculate_arc_length(string):
    nnodes = string.shape[0]
    L = np.zeros((nnodes,))
    s = np.zeros((nnodes,))
    for i in range(1, nnodes):
        L[i] = magnitude((string[i] - string[i - 1]).flatten())
        s[i] = s[i - 1] + L[i]
    return s


def project_trans_rot(a, b):
    centroid_a = np.mean(a, axis=0, keepdims=True)
    centroid_b = np.mean(b, axis=0, keepdims=True)
    A = a - centroid_a
    B = b - centroid_b
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = centroid_b @ R - centroid_a
    return a.flatten(), (b @ R - t).flatten()


def generate_inertia_I(X):
    I = np.zeros((3, 3))
    I[0, 0] = np.sum(X[:, 1] ** 2 + X[:, 2] ** 2)
    I[1, 1] = np.sum(X[:, 0] ** 2 + X[:, 2] ** 2)
    I[2, 2] = np.sum(X[:, 0] ** 2 + X[:, 1] ** 2)
    I[0, 1] = -np.sum(X[:, 0] * X[:, 1])
    I[0, 2] = -np.sum(X[:, 0] * X[:, 2])
    I[1, 2] = -np.sum(X[:, 1] * X[:, 2])
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I


def generate_project_rt(X):
    N = X.shape[0]
    I = generate_inertia_I(X)
    evals, evecs = scipy.linalg.eigh(I)
    evecs = evecs.T

    # use the convention that the element with largest abs. value
    # within a given evec should be positive
    for i in range(3):
        if np.max(evecs[i]) < np.abs(np.min(evecs[i])):
            evecs[i] *= -1

    P = X @ evecs

    R_rot1 = np.zeros(N * 3)
    R_rot2 = np.zeros(N * 3)
    R_rot3 = np.zeros(N * 3)
    R_x = np.zeros(N * 3)
    R_y = np.zeros(N * 3)
    R_z = np.zeros(N * 3)

    for i in range(N):
        for j in range(3):
            R_rot1[3 * i + j] = P[i, 1] * evecs[j, 2] - P[i, 2] * evecs[j, 1]
            R_rot2[3 * i + j] = P[i, 2] * evecs[j, 0] - P[i, 0] * evecs[j, 2]
            R_rot3[3 * i + j] = P[i, 0] * evecs[j, 1] - P[i, 1] * evecs[j, 0]

    for i in range(N):
        R_x[3 * i] = 1.0
        R_y[3 * i + 1] = 1.0
        R_z[3 * i + 2] = 1.0

    R_x = normalize(R_x)
    R_y = normalize(R_y)
    R_z = normalize(R_z)
    R_rot1 = normalize(R_rot1)
    R_rot2 = normalize(R_rot2)
    R_rot3 = normalize(R_rot3)

    proj = np.eye(N * 3)
    proj -= np.outer(R_x, R_x)
    proj -= np.outer(R_y, R_y)
    proj -= np.outer(R_z, R_z)
    proj -= np.outer(R_rot1, R_rot1)
    proj -= np.outer(R_rot2, R_rot2)
    proj -= np.outer(R_rot3, R_rot3)
    return proj


def generate_project_rt_tan(structure, tangent):
    proj = generate_project_rt(structure)
    proj_tangent = proj @ tangent
    proj_tangent = normalize(proj_tangent)
    proj -= np.outer(proj_tangent, proj_tangent)
    return proj
