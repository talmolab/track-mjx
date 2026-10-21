"""Uniformly rescale a STAC reference-clip HDF5 (positions only; angles are invariant).

The MIMIC-MJX release file ``celegans_ik_only_04182019am_centerline_locomotion_2d.h5``
was fit at SCALE_FACTOR 1.0 on a worm whose segment spacing is 10.0x that of the
vnl-playground ``celegans.xml`` model (0.0419 m vs 0.0042 m). Every imitation episode
therefore terminates at step 1 with ``root_too_far``. The paper checkpoint was trained on
a ``*_scale-0.1.h5`` variant; this script regenerates it.

Usage: python scripts/release_checkpoints/scale_worm_reference.py <in.h5> <out.h5> [--scale 0.1]
"""
import argparse, re
import h5py, numpy as np

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst"); p.add_argument("--scale", type=float, default=0.1)
p.add_argument("--n_root_trans", type=int, default=3, help="leading translational root DOFs in qpos/qvel")
a = p.parse_args()

with h5py.File(a.src, "r") as f, h5py.File(a.dst, "w") as g:
    for k in f.keys():
        v = f[k][()]
        if k in ("xpos", "kp_data", "marker_sites", "offsets"):
            v = v * a.scale                                   # all are Cartesian positions
        elif k in ("qpos", "qvel"):
            v = v.copy(); v[:, : a.n_root_trans] *= a.scale   # root translation / velocity only
        elif k == "config":
            s = v.decode() if isinstance(v, bytes) else str(v)
            s = re.sub(r"(SCALE_FACTOR:\s*)[0-9.]+", rf"\g<1>{a.scale}", s)          # SCALE_FACTOR and MOCAP_SCALE_FACTOR
            v = s.encode()
        g.create_dataset(k, data=v)
    print("wrote", a.dst, "| root x range", g["qpos"][:, 0].min(), g["qpos"][:, 0].max(),
          "| body extent frame0 %.4f m" % np.linalg.norm(g["xpos"][0].max(0) - g["xpos"][0].min(0)))
