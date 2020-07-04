CUDA code of our paper:

*Accelerating Translational Image Registration for HDR Images on GPU*

## Commands

_Note: Requires CUDA to be installed on the machine._

To compile:

```
nvcc Project.cu -O3
```

To run:

```
./a.out input/1.JPG input/2.JPG input/3.JPG input/4.JPG input/5.JPG
```
## Credit

STB Image Library (used to read/write images from/to the disk):

*https://github.com/nothings/stb*
