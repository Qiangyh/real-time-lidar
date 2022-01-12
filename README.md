# real-time-sp-lidar

Package containing C++/CUDA algorithms for real-time reconstruction of single-photon lidar data

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
