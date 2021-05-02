import logging
from typing import Dict

import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
from skimage import morphology as morph

from cliodynamics.models.frontier_attacks import compute_attacks


INT_t = np.int64
FLOAT_t = np.float64


class MetaethnicFrontierModel:
    """Turchin's metaethnic frontier model on a square grid.

    Attributes
    ----------
    """
    _CONNECTIVITY = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]], dtype=np.uint8
    )

    def __init__(self,
                 membership: np.ndarray,
                 asabiya: np.ndarray,
                 r0: float = 0.2,
                 delta: float = 0.1,
                 h: float = 2.0,
                 delta_p: float = 0.1,
                 s_crit: float = 0.003):

        if membership.shape != asabiya.shape:
            raise ValueError("Unequal argument shapes.")

        if asabiya.min() < 0 or asabiya.max() > 1:
            raise ValueError("Asabiya value out of bounds.")

        if membership.min() < 0:
            raise ValueError("Memberships must be >= 0.")

        # Model state
        self._membership = membership.astype(INT_t)
        self._asabiya = asabiya.astype(FLOAT_t)
        self._max_empire_id = 1

        # Model parameters
        self._r0 = r0
        self._delta = delta
        self._h = h
        self._delta_p = delta_p
        self._s_crit = s_crit

    @classmethod
    def empty_model(cls, size: int, max_history: int = 100):
        membership = np.zeros((size, size), dtype=INT_t)
        asabiya = np.zeros((size, size), dtype=FLOAT_t)

        # Sensible defaults according to Turchin
        start = np.random.randint(0, size - 4)
        membership[start:start + 4, start:start + 4] = 1
        asabiya[:] = 0.25

        return cls(
            membership=membership,
            asabiya=asabiya,
        )

    @property
    def membership(self) -> np.ndarray:
        return self._membership

    @property
    def asabiya(self) -> np.ndarray:
        return self._asabiya

    def get_empires(self) -> np.ndarray:
        empires = np.unique(self._membership)
        empires = empires[np.nonzero(empires)]
        assert 0 not in empires
        return empires

    def get_areas(self) -> Dict[int, float]:
        empires = self.get_empires()
        areas = np.sum(
            np.array(
                empires[:, np.newaxis, np.newaxis] ==
                self._membership[np.newaxis, :, :]
            ).astype(FLOAT_t), axis=(1, 2),
        )

        assert areas.shape == (len(empires),)
        return dict(zip(empires, areas))

    def step(self):
        ## Update asabiya map
        self._update_asabiya()

        ## Area of control for each empire
        areas_dense = self._get_empire_areas()
        
        ## Average asabiya for each empire
        average_asabiyas_dense = self._get_empire_asabiyas()

        ## Compute distances of each cell to its imperial center
        distances = self._get_empire_distances_from_center()

        ## Compute power for each cell
        powers = areas_dense * average_asabiyas_dense * \
            np.exp(- distances / self._h)

        ## Create a visitation schedule for each interior cell
        schedule = self._create_schedule()

        self._max_empire_id = compute_attacks(
            schedule=schedule,
            powers=powers,
            membership=self._membership,
            asabiya=self._asabiya,
            max_empire_id=self._max_empire_id,
            delta_p=self._delta_p,
        )

        ## Average asabiya for each empire, one more time (post-attack)
        average_asabiyas_dense = self._get_empire_asabiyas()

        ## Check for imperial collapse
        collapsed = np.logical_and(
            average_asabiyas_dense < self._s_crit,
            self._membership != 0,
        )
        self._membership[collapsed] = 0

        ## Edge of the map
        self._update_edge_cells()

        return powers

    def _update_asabiya(self):
        dilated = morph.dilation(self._membership, selem=self._CONNECTIVITY)
        eroded = morph.erosion(self._membership, selem=self._CONNECTIVITY)
        boundary = np.logical_or(
            self._membership != dilated,
            self._membership != eroded,
        )
        
        # Updates on boundary pixels
        self._asabiya[boundary] += self._r0 * self._asabiya[boundary] * \
            (1 - self._asabiya[boundary])

        # Updates on interior pixels
        self._asabiya[~boundary] *= (1 - self._delta)

    def _get_empire_areas(self):
        empire_to_area = self.get_areas()

        # Convert to dense tensor
        areas_dense = np.ones_like(self._membership, dtype=FLOAT_t)
        for eid, area in empire_to_area.items():
            if eid == 0:
                # "No empire" means area is one
                continue

            mask = self._membership == eid
            areas_dense[mask] = area

        return areas_dense

    def _get_empire_asabiyas(self):
        empires = self.get_empires()
        average_asabiyas = ndi.mean(
            input=self._asabiya,
            labels=self._membership,
            index=empires,
        )

        assert average_asabiyas.shape == (len(empires),)

        # Convert to dense tensor
        average_asabiyas_dense = self._asabiya.copy()
        for eid, asabiya in zip(empires, average_asabiyas):
            if eid == 0:
                # "No empire" means asabiya is already its own average
                continue

            mask = self._membership == eid
            average_asabiyas_dense[mask] = asabiya

        return average_asabiyas_dense

    def _get_empire_distances_from_center(self):
        empires = self.get_empires()
        centers = np.array(ndi.center_of_mass(
            input=np.ones_like(self._membership),
            labels=self._membership,
            index=empires,
        ), dtype=FLOAT_t)

        assert centers.shape == (len(empires), 2)

        distances = np.zeros_like(self._membership, dtype=FLOAT_t)
        for eid, center in zip(empires, centers):
            if eid == 0:
                # "No empire" means distance to self is zero
                continue

            mask = self._membership == eid
            ijs = np.array(np.nonzero(mask), dtype=FLOAT_t).T

            assert ijs.ndim == 2
            assert ijs.shape[1] == 2

            distances[mask] = np.linalg.norm(ijs - center, axis=1)

        return distances

    def _create_schedule(self):
        height, width = self._membership.shape

        # Get list of coordinates as (N, 2)-shaped arrays
        ijs = np.mgrid[1:height - 1, 1:width - 1].T.reshape(-1, 2)
        ijs_north = np.mgrid[:height - 2, 1:width - 1].T.reshape(-1, 2)
        ijs_south = np.mgrid[2:height, 1:width - 1].T.reshape(-1, 2)
        ijs_west = np.mgrid[1:height - 1, :width - 2].T.reshape(-1, 2)
        ijs_east = np.mgrid[1:height - 1, 2:width].T.reshape(-1, 2)

        assert ijs.shape == ((height - 2) * (width - 2), 2)
        assert ijs_north.shape == ((height - 2) * (width - 2), 2)
        assert ijs_south.shape == ((height - 2) * (width - 2), 2)
        assert ijs_west.shape == ((height - 2) * (width - 2), 2)
        assert ijs_east.shape == ((height - 2) * (width - 2), 2)

        # The schedule is (M, 4)-shaped where each row contains the
        # two coordinates of the attacker followed by the two of the
        # defender. Attacker coordinates are repeated across rows.
        schedule = np.vstack((
            np.hstack((ijs, ijs_north)),
            np.hstack((ijs, ijs_south)),
            np.hstack((ijs, ijs_west)),
            np.hstack((ijs, ijs_east)),
        ))

        assert schedule.ndim == 2
        assert schedule.shape[0] == (height - 2) * (width - 2) * 4
        assert schedule.shape[1] == 4

        # Randomize schedule
        np.random.shuffle(schedule)

        return schedule

    def _update_edge_cells(self):
        self._asabiya[:, 0] = self._asabiya[:, 1]
        self._membership[:, 0] = self._membership[:, 1]
        self._asabiya[:, -1] = self._asabiya[:, -2]
        self._membership[:, -1] = self._membership[:, -2]
        self._asabiya[0, :] = self._asabiya[1, :]
        self._membership[0, :] = self._membership[1, :]
        self._asabiya[-1, :] = self._asabiya[-2, :]
        self._membership[-1, :] = self._membership[-2, :]
