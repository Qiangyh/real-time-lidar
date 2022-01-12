# Real-time single-photon lidar package

This package containing C++/CUDA algorithms for real-time reconstruction of single-photon lidar data

## Algorithms

All the algorithms in this package can achieve real-time performance.

### Global (spatial regularisation)

- [RT3D](#rt3d)
- [Color RT3D](#color-rt3d)
- [Sketched RT3D](#sketched-rt3d)

### Pixelwise (no regularisation)

- Log-matched filtering
- Matched filtering
- [Half-sample mode](#color-rt3d)
- [Sketched maximum likelihood](#sketched-rt3d)

## Dependencies

- Point Cloud Library 1.8.1 (visualisation/point primitives)
- Eigen3 (computing primitives)
- Ponca (point cloud fitting)
- FreeImage (visualization)

## Compilation

```bash
git clone https://gitlab.com/Tachella/real-time-sp-lidar.git
cd real-time-sp-lidar
git submodule update --init --recursive
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE
make
```

## Papers

### RT3D
Full paper: [https://www.nature.com/articles/s41467-019-12943-7](https://www.nature.com/articles/s41467-019-12943-7)

```
    @article{tachella2019rt3d,
    title={Real-time 3{D} reconstruction from single-photon 
    lidar data using plug-and-play point cloud denoisers},
    author={Tachella, J. and Altmann, Y. and Mellado, N. and McCarthy,
     A. and Tobin, R. and Buller, G. S. and Tourneret, J.Y. and McLaughlin, S.},
    journal={Nature communications},
    volume={10},
    number={1},
    year={2019}
    }
```

### Color RT3D 

Full paper: [https://ieeexplore.ieee.org/abstract/document/9022496](https://ieeexplore.ieee.org/abstract/document/9022496)
```
@inproceedings{tachella2019crt3d,
author={Tachella, J. and Altmann, Y. 
and McLaughlin, S. and Tourneret, J.-Y.},
booktitle={2019 IEEE 8th International Workshop
 on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP)}, 
title={Real-Time 3D Color Imaging with Single-Photon Lidar Data}, 
year={2019},
volume={},
number={},
pages={206-210},
doi={10.1109/CAMSAP45676.2019.9022496}} 
```

### Sketched RT3D
Full paper: [https://ieeexplore.ieee.org/document/9616208](https://ieeexplore.ieee.org/document/9616208)

```
@article{sheehan2021sketchedlidar,
author={Sheehan, Michael and Tachella, Julian and Davies, Mike E.},
journal={IEEE Trans. on Comp. Imag.}, 
title={A Sketching Framework for
 Reduced Data Transfer in Photon Counting Lidar}, 
year={2021},
volume={},
number={},
pages={},
doi={10.1109/TCI.2021.3113495}
}
```
