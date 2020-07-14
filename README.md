This repository contains companion code for the following preprint:

K.C. Alpay, K.B. Aydemir, A. Temizel, “Accelerating Translational Image Registration for HDR Images on GPU”, arXiv:2007.046483, July 2020.

https://arxiv.org/abs/2007.06483


If you use this code please cite the paper using the bibtex reference below:

@article{2020arXiv200706483C,

author = {{Cenk Alpay}, Kadir and {Berkay Aydemir}, Kadir and {Temizel}, Alptekin},

title = "{Accelerating Translational Image Registration for HDR Images on GPU}",

journal = {arXiv e-prints},

year = 2020

}


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
## Credits

STB Image Library (used to read/write images from/to the disk):

*https://github.com/nothings/stb*
