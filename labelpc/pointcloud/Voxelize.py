
import numpy as np
from collections import defaultdict
import hdbscan as hdb
from tqdm import trange, tqdm


class VoxelGrid:
    """
    This class implements a simple voxelization technique for sorting a point cloud for fast querying of
    local point neighborhoods. The voxel grid is made of cubical voxels with some given length (mesh_size).
    Each point in the original input point cloud falls into some voxel in the grid.

    Once the voxel grid has been built and the point cloud has been sorted (voxelized), queries can be made to find
    all the points in a given region of space.
    """
    def __init__(self, points, mesh_size=0.02, offset=(0.0, 0.0, 0.0)):
        """
        Initialize a VoxelGrid object with a given mesh size and buffer. This function takes in a point cloud,
        builds the grid, and sorts (voxelizes) the input point cloud so that it is ready for queries. The points
        are NOT stored in this data structure, only their indices are stored and returned in queries.
        :param points: Input point cloud.
        :param mesh_size: Length of each side of the voxels (cubes).
        :param offset: 3D offset value allows to shift the voxel grid by some amount relative to the point cloud.
        This parameter effectively moves the origin of the voxel grid, allowing for more control of the grid.
        """
        self.mesh_size = np.ones(3, dtype=float) * mesh_size
        self.offset = np.array(offset)
        self.points = np.array(points)
        self.n_occupied_voxels = None
        self.voxel_idx, self.sorted_points = self.voxelize()

    def __len__(self):
        """
        The length of a VoxelGrid object is the number of occupied voxels.
        """
        if self.n_occupied_voxels is None:
            self.n_occupied_voxels = 0
            for k in self.sorted_points.keys():
                if isinstance(k, tuple):
                    self.n_occupied_voxels += 1
        return self.n_occupied_voxels

    def index(self, point):
        """
        This function calculates the voxel index (xidx, yidx, zidx) of the input point. Note that the indices could
        be negative if the input point is below or left of the original point cloud
        :param point: Point of interest whose voxel indices to be calculated.
        :return: xidx, yidx, zidx (the 3D indices of the voxel containing the point).
        """
        return tuple(((point + self.offset) // self.mesh_size).astype(int))

    def voxelize(self):
        """
        This function takes in a point cloud and a mesh size, calculates the voxel grid.
        :return: idx, a list of xidx, yidx, zidx tuples for each point.
                 sorted_points, a dictionary mapping voxel indices to lists of point indices contained in the voxel),
                 nbins (total number of bins in the x, y, and z directions [i.e. xbins, ybins, zbins]).
        """
        idx = list(((self.points + self.offset) / self.mesh_size).astype(int))

        # Sort the points into voxels
        sorted_points = defaultdict(list)
        if len(self.points) > 1000000:
            for i in trange(len(self.points), desc='Building voxel grid'):
                idx[i] = tuple(idx[i])
                sorted_points[idx[i]].append(i)
        else:
            for i in range(len(self.points)):
                idx[i] = tuple(idx[i])
                sorted_points[idx[i]].append(i)

        return idx, sorted_points

    def min_corner(self, index=None):
        """
        This function returns the bottom corner of the given voxel.
        :param index: Voxel index (xidx, yidx, zidx).
        :return: min_point (bottom corner of voxel).
        """
        if index is None:
            return np.min(self.points, axis=0)
        else:
            return np.array(index) * self.mesh_size - self.offset

    def max_corner(self, index=None):
        """
        This function returns the top corner of the given voxel.
        :param index: Voxel index (xidx, yidx, zidx).
        :return: max_point (top corner of voxel).
        """
        if index is None:
            return np.max(self.points, axis=0)
        else:
            return self.min_corner(index) + self.mesh_size

    def center(self, index=None):
        """
        This function returns the center of the given voxel.
        :param index: Voxel index (xidx, yidx, zidx).
        :return: center_point (center of voxel).
        """
        if index is None:
            return (self.min_corner() + self.max_corner()) / 2.0
        else:
            return self.min_corner(index) + self.mesh_size / 2.0

    def neighbors(self, center, radius=None, overlap=None):
        """
        This function returns all the voxels that make up a super voxel with the queried point in the center voxel.
        A super voxel is a cube made of voxels. The first parameter can either be a voxel index or a point in space.
        :param center: Voxel index (xidx, yidx, zidx) of the center of the super voxel or a point in space.
        :param radius: Approximate side length of the super voxel (given in units of meters).
        :param overlap: Number of neighboring voxels to use in query.
        :return: List of voxel indices making up the super voxel.
        """
        if not isinstance(center, tuple):
            center = self.index(center)

        if overlap is not None:
            radius = overlap
        elif radius is not None:
            radius = int(radius / self.mesh_size)
        else:
            radius = 1

        super_voxel = []
        for i in range(-radius, radius+1):
            xidx = center[0] + i
            for j in range(-radius, radius+1):
                yidx = center[1] + j
                for k in range(-radius, radius+1):
                    zidx = center[2] + k
                    super_voxel.append((xidx, yidx, zidx))
        return super_voxel

    def fullest(self):
        """
        :return: Index of the voxel which contains the most points. If there are more than one voxels with this
        number of points, only the first index is returned.
        """
        idx = (0, 0, 0)
        max_count = 0
        for key, contents in self.sorted_points.items():
            if len(contents) > max_count:
                max_count = len(contents)
                idx = key
        return idx

    def counts(self, voxels=None):
        """
        Return the number of points in each of the selected voxels in the form of a dictionary using the voxel
        index as a key. If no voxels are selected, return counts for all occupied voxels. If only a single
        voxel is queried, just return an integer.
        """
        if voxels is None:
            voxels = self.occupied()
        elif not len(voxels):
            return []
        elif not isinstance(voxels[0], tuple):
            return len(self.sorted_points[voxels])

        counts = defaultdict(lambda: 0)
        for v in voxels:
            counts[v] = len(self.sorted_points[v])
        return counts

    def indices(self, voxels=None, merge=False):
        """
        Get a total list of indices of all the points in all the selected voxels. If no voxels are selected, get
        all the point indices from all occupied voxels. If merge is True, merge everything into one list.
        """
        if voxels is None:
            if merge:
                return [i for v in self.occupied() for i in self.sorted_points[v]]
            else:
                return [self.sorted_points[v] for v in self.occupied()]
        elif not len(voxels):
            return []
        elif isinstance(voxels, tuple) and not isinstance(voxels[0], tuple):
            return self.sorted_points[voxels]

    def contents(self, voxels=None, merge=False):
        """
        Get the points contained in all the selected voxels. If no voxels are selected, get the points from all
        occupied voxels (separated by voxel). If merge is True, merge all points into one list.
        """
        if voxels is None:
            if merge:
                return self.points
            else:
                return [self.points[indices] for indices in self.indices()]
        elif not len(voxels):
            return []
        elif isinstance(voxels, tuple) and not isinstance(voxels[0], tuple):
            return self.points[self.sorted_points[voxels]]

    def occupied(self):
        """
        Return a list of voxel indices of all non-empty voxels.
        """
        return [idx for idx in self.sorted_points.keys() if len(self.sorted_points[idx])]

    def all(self):
        """
        Return a list of all voxel indices including empty voxels that span from the bottom left-most occupied voxel to
        the upper right-most occupied voxel.
        """
        min_idx = self.index(self.min_corner())
        max_idx = self.index(self.max_corner())
        return [(i, j, k) for i in range(min_idx[0], max_idx[0]+1) for j in range(min_idx[1], max_idx[1]+1) for k in range(min_idx[2], max_idx[2]+1)]

    def bitmapFromSlice(self, max=None, scores=None, min_idx=None, max_idx=None):
        """
        Return an array of integers indicating the number of points in each voxel in the grid. This array is
        3-dimensional, spanning all the voxels in the grid. If max is not None, then adjust the values in the
        bitmap to have the largest value equal to max. This bitmap can be fed directly into pyqtgraph.image() function.
        If the user passes per-point scores into this function, then those scores will be used to create the bitmap,
        otherwise the value will be set to the max value for all occupied voxels.
        """
        if min_idx is None:
            min_idx = np.array(self.index(self.min_corner()))
        if max_idx is None:
            max_idx = np.array(self.index(self.max_corner()))
        range_idx = max_idx - min_idx + 1
        bitmap = np.zeros(range_idx[:2], dtype=int)
        if scores is None:
            for v in self.occupied():
                i, j, _ = np.array(v) - min_idx
                #bitmap[i][range_idx[1] - j - 1][k] = self.counts(v)
                bitmap[i][range_idx[1] - j - 1] = max
        else:
            for v in self.occupied():
                i, j, k = np.array(v) - min_idx
                bitmap[i][range_idx[1] - j - 1] = np.average(scores[self.indices(v)])
        return np.swapaxes(bitmap, 0, 1)

    def bitmap3d(self, max=None, swapaxes=True, scores=None):
        """
        Return an array of integers indicating the number of points in each voxel in the grid. This array is
        3-dimensional, spanning all the voxels in the grid. If max is not None, then adjust the values in the
        bitmap to have the largest value equal to max. This bitmap can be fed directly into pyqtgraph.image() function.
        If the user passes per-point scores into this function, then those scores will be used to create the bitmap,
        otherwise the value will be set to the max value for all occupied voxels.
        """
        min_idx = np.array(self.index(self.min_corner()))
        max_idx = np.array(self.index(self.max_corner()))
        range_idx = max_idx - min_idx + 2
        bitmap = np.zeros(range_idx, dtype=int)
        if scores is None:
            for v in tqdm(self.occupied(), desc='Building 3D bitmap'):
                i, j, k = np.array(v) - min_idx
                #bitmap[i][range_idx[1] - j - 1][k] = self.counts(v)
                bitmap[i][range_idx[1] - j - 1][k] = max
        else:
            for v in tqdm(self.occupied(), desc='Building 3D bitmap'):
                i, j, k = np.array(v) - min_idx
                bitmap[i][range_idx[1] - j - 1][k] = np.average(scores[self.indices(v)])
        if max is not None:
            bitmap = (bitmap * max / bitmap.max()).astype(int)
        if swapaxes:
            return np.swapaxes(bitmap, 0, 2)
        else:
            return bitmap

    def bitmap2d(self, max=None, axis=2, scores=None):
        map = self.bitmap3d(max, scores=scores)
        maps = []
        for i in range(map.shape[2-axis]):
            if axis == 0:
                maps.append(map[:, :, i])
            elif axis == 1:
                maps.append(map[:, i, :])
            elif axis == 2:
                maps.append(map[i, :, :])
            else:
                print('Invalid axis choice')
                return []
        return maps

    def strip(self, center, radius=-1.0, axis=2):
        """
        Return a list of voxel indices of a strip of voxels along one of the axes.
        :param center: Voxel index (xidx, yidx, zidx) of the center voxel.
        :param radius: If greater than zero, return a strip of length approximately 2 * radius.
        :param axis: Choose which axis to take the strip along (x,y,z --> 0,1,2).
        :return: List of voxel indices of the voxels in the strip.
        """
        strip = []
        radius = radius // self.mesh_size
        min_pos, max_pos = center[axis] - radius, center[axis] + radius
        for key in self.sorted_points.keys():
            if radius[0] < 0.0 or min_pos <= key[axis] <= max_pos:
                if key[axis-2] == center[axis-2] and key[axis-1] == center[axis-1]:
                    strip.append(key)
        return strip

    def cluster(self, voxels=None, return_noise=False, min_cluster_size=30, min_samples=5):
        """
        Cluster the given voxel in the given grid using HDBSCAN and return the clustered point indices. If no
        voxel index is given, perform clustering for all occupied voxels. If noise is True, then return the
        points that were not part of any cluster as a separate array.
        """
        if voxels is None:
            voxels = self.occupied()
        elif not len(voxels):
            return []
        elif not isinstance(voxels[0], tuple):
            voxels = [voxels]

        count = 0
        clusters = []
        noise = []
        not_enough = 0
        for voxel in voxels:
            count += 1
            indices = np.array(self.indices(voxel))
            # If we have at least 50 points in the voxel, then cluster the points
            if len(indices) > 50:
                if not count % 10:
                    print('working on voxel ', voxel, '[%d/%d]' % (count, len(voxels)))
                clusterer = hdb.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                labels = clusterer.fit_predict(self.points[indices])
                # Record which points are in which cluster
                if return_noise:
                    noise.extend(indices[labels == -1])
                for l in range(0, np.max(labels) + 1):
                    clusters.append(indices[labels == l])
            else:
                not_enough += 1

        print(not_enough, 'out of', len(voxels), 'voxels didn\'t have enough points to be clustered')

        if return_noise:
            return clusters, noise
        else:
            return clusters

    def curvature(self, neighbors=0):
        """
        Calculate the curvature for every point in the input point cloud using the voxel grid as a way to
        identify neighbors. If neighbors is equal to 2, then the two voxels to the left, right, up, down, etc
        directions will be used for the computation around each voxel.
        """
        curvature = np.empty(len(self.points))
        for voxel in self.occupied():
            if neighbors:
                indices = self.indices(self.neighbors(voxel, overlap=neighbors), merge=True)
            else:
                indices = self.indices(voxel)

            eigen = feat.eigenvalues_single(self.points[indices], self.center(voxel))
            if eigen[0] > 0.0:
                curvature[indices] = 3.0 * eigen[0] / np.sum(eigen)
            else:
                curvature[indices] = 0.0

        return curvature

