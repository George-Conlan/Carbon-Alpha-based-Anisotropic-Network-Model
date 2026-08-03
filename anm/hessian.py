from scipy.sparse import lil_matrix

import numpy as np
def pair_block(coord_i, coord_j, gamma):
    diff = coord_j - coord_i
    dist = np.linalg.norm(diff)
    u = diff / dist
    block = -gamma * np.outer(u, u)
    return block


def build_hessian(coords, contacts, gamma):
    matrix_size = 3*coords.shape[0]
    matrix = lil_matrix((matrix_size, matrix_size))
    for i,j in contacts:
        block = pair_block(coords[i], coords[j], gamma)
        matrix[3*i:3*i+3, 3*j:3*j+3] += block
        matrix[3*j:3*j+3, 3*i:3*i+3] += block.T
        matrix[3*i:3*i+3, 3*i:3*i+3] -= block
        matrix[3*j:3*j+3, 3*j:3*j+3] -= block

    return matrix.tocsr()
