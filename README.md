This repository contains companion code for the following preprint:


*K.C. Alpay, K.B. Aydemir, A. Temizel, “Accelerating Translational Image Registration for HDR Images on GPU”, arXiv:2007.06483, July 2020.*


https://arxiv.org/abs/2007.06483


<img src="https://kadircenk.github.io/img/blurryartifact.png" width="500" height="auto">


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

> Alpay, K. C., Aydemir, K. B., & Temizel, A. (2020). Accelerating Translational Image Registration for HDR Images on GPU. arXiv preprint arXiv:2007.06483.

BibTeX entry:

```
@article{alpay2020accelerating,
  title={Accelerating Translational Image Registration for HDR Images on GPU},
  author={Alpay, Kadir Cenk and Aydemir, Kadir Berkay and Temizel, Alptekin},
  journal={arXiv preprint arXiv:2007.06483},
  year={2020}
}
```



## Credits

STB Image Library (used to read/write images from/to the disk):

*https://github.com/nothings/stb*
