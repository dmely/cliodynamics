import numpy as np

cimport numpy as np
cimport cython
np.import_array()

INT_t = np.int64
FLOAT_t = np.float64


def compute_attacks(np.ndarray schedule,
                    np.ndarray powers,
                    np.ndarray membership,
                    np.ndarray asabiya,
                    int max_empire_id,
                    float delta_p):
    
    cdef int height = powers.shape[0]
    cdef int width = powers.shape[1]
    cdef int schedule_length = schedule.shape[0]

    # Assertions
    assert schedule.dtype == INT_t
    assert powers.dtype == FLOAT_t
    assert membership.dtype == INT_t
    assert asabiya.dtype == FLOAT_t

    assert powers.ndim == 2
    assert membership.ndim == 2
    assert asabiya.ndim == 2

    cdef int m_h = membership.shape[0]
    cdef int m_w = membership.shape[1]
    cdef int a_h = asabiya.shape[0]
    cdef int a_w = asabiya.shape[1]
    assert height == m_h
    assert width == m_w
    assert height == a_h
    assert width == a_w

    assert schedule_length == (height - 2) * (width - 2) * 4
    assert schedule.shape[1] == 4

    # Iterate
    cdef int s, i1, j1, i2, j2, num_empires
    cdef float p_attacker, p_defender
    num_empires = max_empire_id

    for s in range(schedule_length):
        i1 = schedule[s, 0]
        j1 = schedule[s, 1]
        i2 = schedule[s, 2]
        j2 = schedule[s, 3]

        p_attacker = powers[i1, j1]
        p_defender = powers[i2, j2]

        if p_attacker - p_defender > delta_p:
            if membership[i1, j1] == 0: # New empire!
                num_empires += 1
                membership[i1, j1] = num_empires

            membership[i2, j2] = membership[i1, j1]
            asabiya[i2, j2] = (asabiya[i2, j2] + asabiya[i1, j1]) / 2.

    return num_empires