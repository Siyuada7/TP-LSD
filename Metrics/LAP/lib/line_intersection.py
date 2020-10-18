import numpy as np

def line_area_intersection(l_src, l_tar):
    """
    calculate the overlapping area
    :param l_src: gt(N)
    :param l_tar: est(1)
    :return: est-> overlapping area to each gt
    """
    N =len(l_src)
    idx_valid = [False for i in range(N)]
    pd_covered = np.zeros(N)
    vec_base = l_src[:, 2:4] - l_src[:, 0:2]
    vec_base = vec_base / np.expand_dims(np.linalg.norm(vec_base, axis=1), axis=1).repeat(2, axis=1)
    vec_src = np.array([[(l_src[i, 0:2]-l_src[i, 0:2]).dot(vec_base[i].transpose()),(l_src[i,2:4]-l_src[i,0:2]).dot(vec_base[i].transpose())] for i in range(N)])

    index = np.where(vec_src[:, 0] > vec_src[:,1])[0]
    a, b = [0, 1], [1, 0]
    for index_i in range(len(index)):
        vec_src[index[index_i], a] = vec_src[index_i, b]

    vec_tar = np.array([[(l_tar[0:2]-l_src[i, 0:2]).dot(vec_base[i].transpose()),(l_tar[2:4]-l_src[i,0:2]).dot(vec_base[i].transpose())] for i in range(N)])
    index = np.where(vec_tar[:, 0] > vec_tar[:, 1])[0]
    for index_i in range(len(index)):
        vec_tar[index[index_i], a] = vec_tar[index[index_i], b]

    # clip left area
    idx = np.where(vec_tar.reshape(-1) < 0)[0]
    if len(idx) > 0:
        vec_tar.reshape(-1)[idx] = 0
    # clip right area
    for i in range(len(vec_tar)):
        idx = np.where(vec_tar[i] > np.max(vec_src[i]))[0]
        if len(idx) > 0:
            vec_tar[i, idx] = vec_src[i].max()

    # overlapping
    for k in range(N):
        bValid = True
        if vec_tar[k, 0] >= vec_src[k,1] or vec_tar[k, 1] <= vec_src[k, 0]:
            # case1
            # tar:                    *---------- *
            # src: *--------- *
            # case2
            # tar: *---------- *
            # src:                 *--------- *
            idx_valid[k] = False
            pd_covered[k] = 0
            bValid = False
        elif vec_tar[k, 0] <= vec_src[k, 0] and vec_tar[k, 1] >= vec_src[k, 0] and vec_tar[k,1] <= vec_src[k, 1]:
            #case 3
            #tar: *-------*
            #src:    *---------*
            vec_tar[k, 0] = vec_src[k, 0]
        elif vec_tar[k, 0] >= vec_src[k, 0] and vec_tar[k, 0] <= vec_src[k, 1] and vec_tar[k,1] >= vec_src[k, 1]:
            #case 4
            #tar:       *-------*
            #src: *---------*
            vec_tar[k, 1] = vec_src[k, 1]
        elif vec_tar[k, 0] <= vec_src[k, 0] and vec_tar[k,1] >= vec_src[k, 1]:
            #case 5
            #tar:   *-------------*
            #src:     *---------*
            vec_tar[k, :] = vec_src[k, :]
        else:
            # case 5
            # tar:   *----------*
            # src: *-------------*
            pass
        if bValid:
            idx_valid[k] = True
            pd_covered[k] = abs(vec_tar[k, 1] - vec_tar[k, 0])
    #print(idx_valid, pd_covered)
    return idx_valid, pd_covered


# l_src = np.array([[50, 50, 100, 50], [50, 46, 100, 57]])
# l_tar = np.array([50, 50, 100, 50])
# line_area_intersection(l_src, l_tar)