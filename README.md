# Visual-Hull

**Visual-Hull** is a CUDA implementation of voxel-based visual hull algorithm. It takes four binary images (usually the output of motion detection) as input, and gives the moving people's positions as output. You can see [1] and [2] for more details. **Visual-Hull** can be combined with **Particle-Filter-Tracker** together to detect, locate, and track multi-pedestrains in 3-dememsional space by adding a data assocuiation process called trajectory management (my notation), where I call **Visual-Hull** as **localizer**. You can find more results on the homepage of [**[Multi-view pedestrian tracking][homepage]**]. `./matlab/` contains a MATLAB implementation. **POM** (Probabilistic Occupancy Map) [4] is another kind of localizers.

# Some results

A visual hull reconstruction result is shown as follows, the dataset is taken from [4]:

**input** (one view)

![Oops! I cannot find the image!](/images/i.jpg)

**output** (the position can be found by projecting the visual hull into the ground)

![Oops! I cannot find the image!](/images/o.jpg)

# References

[1] POSSEGGER H., STERNIG S., MAUTHNER T., et al. Robust real-time tracking of multiple objects by volumetric mass densities [C]. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013: 2395 ∼ 2402. 

[2] ANDERSEN M., ANDERSEN R. S., KATSARAKIS N., et al. Three-dimensional adaptive sensing of people in a multi-camera setup [C]. The European Signal Processing
Conference, 2010: 964 ∼ 968.

[3] http://cvlab.epfl.ch/software/pom

[4] http://lrs.icg.tugraz.at/download.php




[homepage]: http://zhaozj89.github.io/Visual-Hull/
