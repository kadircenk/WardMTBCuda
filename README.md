This repository contains companion code for the following preprint:


*K.C. Alpay, K.B. Aydemir, A. Temizel, “Accelerating Translational Image Registration for HDR Images on GPU”, arXiv:2007.046483, July 2020.*


https://arxiv.org/abs/2007.06483


<img src="https://kadircenk.github.io/img/blurryartifact.png" width="500" height="490">


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


## Citation

If you use this code, please cite the paper using the reference below:

> K.C. Alpay, K.B. Aydemir, A. Temizel, “Accelerating Translational Image Registration for HDR Images on GPU”, arXiv:2007.046483, July 2020.

BibTeX entry:

```
@article{wardmtbcuda,
title = {Accelerating Translational Image Registration for HDR Images on GPU},
author = {Kadir Cenk Alpay and Kadir Berkay Aydemir and Alptekin Temizel},
journal = {arXiv e-prints arXiv:2007.06483},
year = {2020},
}
```



## Credits

STB Image Library (used to read/write images from/to the disk):

*https://github.com/nothings/stb*
