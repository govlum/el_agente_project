#Takes in a folder containing any amount of xyz files
import os
import numpy as np
from shutil import *
import pandas as pd
from typing import Tuple
#from create_crest_scripts import *
from shutil import *


def transform_numpy_array(mol_coords):
    #Takes in all mol lines. Returns the list of atoms, and a numpy array of coordinates
    coord_lines = mol_coords[2:]
    coords = []
    chars = []
    for line in coord_lines:
        line_arr = line.split()
        coords.append([line_arr[1], line_arr[2], line_arr[3]])
        chars.append(line_arr[0])
    coords = np.array(coords)
    coords = coords.astype(float, copy=False)
    return chars, coords

def read_xyz_file(fname):
    f = open(fname,'r')
    all_lines = f.readlines()
    f.close()
    return all_lines

# ─────────────────────────────  geometry helpers  ──────────────────────────────
def _min_interatomic_distance(xyz_a, xyz_b):
    diff = xyz_a[:, None, :] - xyz_b[None, :, :]
    return float(np.min(np.linalg.norm(diff, axis=-1)))


def _centroid(xyz):
    return xyz.mean(axis=0)


def _rotation_matrix(axis, angle_rad):
    """
    Rodrigues’ rotation formula for unit *axis* and right-handed *angle_rad*.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle_rad / 2.0)
    b, c, d = -axis * np.sin(angle_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array(
        [[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
         [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
         [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]]
    )


def _random_rotation(rng):
    """
    Uniform random 3-D rotation matrix.
    """
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )  # (x, y, z, w) quaternion
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array(
        [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ]
    )


def _apply_rotation(xyz, R):
    cen = _centroid(xyz)
    return (xyz - cen) @ R.T + cen


def _translate_until_clear(
    xyz_mol: np.ndarray,
    xyz_ref: np.ndarray,
    direction: np.ndarray,
    start_shift: float,
    target: float,
    tol: float = 1e-4,
    max_iter: int = 25,
):
    """
    Translate *xyz_mol* along *direction* until every inter-atomic distance to
    *xyz_ref* is ≥ target-tol.  Returns the translated coordinates and the
    final shift (Å) that was applied.
    """
    shift = start_shift
    xyz = xyz_mol + shift * direction
    d_min = _min_interatomic_distance(xyz, xyz_ref)

    it = 0
    while d_min < target - tol and it < max_iter:
        shift += (target - d_min) + tol
        xyz = xyz_mol + shift * direction
        d_min = _min_interatomic_distance(xyz, xyz_ref)
        it += 1

    if d_min < target - tol:
        raise RuntimeError(
            "Unable to meet target separation with translations alone."
        )
    return xyz, shift

def _orient_and_place(
    xyz_template: np.ndarray,
    xyz_ref: np.ndarray,
    direction: np.ndarray,
    start_shift: float,
    target: float,
    rng: np.random.Generator,
    max_rot_trials: int = 100,
) -> np.ndarray:
    """
    Try random rotations (about centroid) followed by translation along
    *direction* until one orientation satisfies the clearance criterion.
    """
    direction = direction / np.linalg.norm(direction)
    for _ in range(max_rot_trials):
        R = _random_rotation(rng)
        xyz_rot = _apply_rotation(xyz_template, R)
        try:
            xyz_pl, _ = _translate_until_clear(
                xyz_rot, xyz_ref, direction, start_shift, target
            )
        except RuntimeError:
            continue  # this rotation failed—try another
        else:
            return xyz_pl

    raise RuntimeError(
        "Exhausted all rotation trials; could not place molecule without clash."
    )

def sandwich_molecules(
    xyz1,
    xyz2,
    target = 3.5,
    max_target = 6.0,
    increment = 0.5,
    *,
    seed = None):
    """
    Place **one rotated & translated copies** of *mol1* around *mol2*
    so that *every* atom pair distance is ≥ `target` Å.

    mol2 will be a 'collector' molecule that will accumulate molecules that have been successfully placed

    Parameters
    ----------
    xyz1, xyz2 : (N1, 3) and (N2, 3) float arrays
        Cartesian coordinates of molecule 1 and molecule 2.
    target : float, optional
        Desired minimum separation (Å) between each copy of mol1 and mol2.
    seed : int or None, optional
        RNG seed for reproducibility of random rotations.

    Returns
    -------
    xyz1_A : (N1, 3) ndarray – first placed copy of mol1
    xyz2   : (N2, 3) ndarray – mol2 (unchanged copy)
    T
    """
    if xyz1.size == 0 or xyz2.size == 0:
        raise ValueError("Input coordinate arrays must be non-empty.")

    rng = np.random.default_rng(seed)

    # ── 0)  resolve complete overlap by translating mol1 ───────────────────────
    diff0 = xyz1[:, None, :] - xyz2[None, :, :]
    dists0 = np.linalg.norm(diff0, axis=-1)
    if np.min(dists0) < 1e-6:
        xyz1 = xyz1 + np.array([0.0, 0.0, 3.5])  # shift mol1 3.5 Å along +z

    while target <= max_target:
        # ── 1)  compute closest atom pair & axis with *current* geometries ────
        diff = xyz1[:, None, :] - xyz2[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        idx1, idx2 = np.unravel_index(int(np.argmin(dists)), dists.shape)
        d_min = dists[idx1, idx2]

        # Axis from the closest atom of mol1 to that of mol2
        v_unit = (xyz2[idx2] - xyz1[idx1]) / d_min

        try:
            xyz1_A = _orient_and_place(
                xyz1,
                xyz2,
                direction=+v_unit,
                start_shift=(target - d_min),
                target=target,
                rng=rng,
            )
        except RuntimeError:
            # placement failed – raise the bar and retry
            target += increment
            continue
        else:
            # success
            print('XYZ 1 A:')
            print(xyz1_A)
            return xyz1_A, xyz2.copy()

    # If the loop exits, even the largest target failed
    raise RuntimeError(
        f"Could not place molecules without clashes even after "
        f"increasing target up to {max_target} Å.")


def create_molfile(chars, coords, name_f):
    new_xyz = ''
    tot_numstr = str(len(coords))
    new_xyz = new_xyz + tot_numstr+'\n'
    new_xyz = new_xyz+'Generated Molecular Structure\n'
    coords = list(coords)
    for i in range(len(coords)):
        coord_str = ''
        coord_str = coord_str + chars[i]+ ' '
        curr_coord = coords[i]
        for num in curr_coord:
            coord_str = coord_str+str(num)+' '
        new_xyz = new_xyz+coord_str+'\n'
    f1 = open(name_f, 'w')
    f1.write(new_xyz)
    f1.close()
    return None


def read_molecules_in_dir(dirname, name_f, target=3.5):
    #Read molecules (xyz files) in a directory specified by dirname. Make sure dirname is full path
    #Returns an xyz file that is an amalgam of all molecules put together
    #Make sure only xyz files are in directory. Feel free to modify this so that it only reads xyz files or whatever
    #Target is target minimum distance between each molecule. Default is 3.5A, feel free to modify it 
    curr_dir = os.getcwd()
    os.chdir(dirname)
    #all_dirs = os.listdir(dirname)
    #First, add xyz1 and xyz2 together to form one xyz file. Then merge the rest with xyz2
    xyz2_coords = np.array([])
    all_fnames = os.listdir(dirname)
    for i in range(len(all_fnames)):
        curr_name = all_fnames[i]
        print(curr_name)
        if i < len(all_fnames) - 1:
            print(all_fnames[i+1])
            if len(xyz2_coords) == 0:
                xyz2_lines = read_xyz_file(all_fnames[0])
                xyz2_chars, xyz2_coords = transform_numpy_array(xyz2_lines)
            xyz1_lines = read_xyz_file(all_fnames[i+1])
            xyz1_chars, xyz1_coords = transform_numpy_array(xyz1_lines)
            xyz1_coords, xyz2_coords = sandwich_molecules(xyz1_coords, xyz2_coords, target)
            #Next, concatenate both coordinates into one 'molecule'
            xyz2_coords = np.array(xyz2_coords.tolist()+xyz1_coords.tolist())
            print('XYZ Concat Coords:')
            print(xyz2_coords)
            xyz2_chars = xyz1_chars + xyz2_chars
    #Return to original directory. Write molfile
    os.chdir(curr_dir)
    create_molfile(xyz2_chars, xyz2_coords, name_f)
    
if __name__ == '__main__':
    curr = os.getcwd()
    read_molecules_in_dir(curr+'/examples','all_molecules_together.xyz')
