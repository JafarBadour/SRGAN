## README
This project was done by Jafar Badour, Hussain Kara Fallah and Almir Mullanurov.

## Description of the project
The goal of Single image super-resolution (SISR) is to recover the lost details from
a low-resolution (LR) image providing a high-resolution (HR) image.

The main issue we face with such task is that there are multiple HR images that could 
produce an LR image upon down-scaling, gaussian noise effects or similar effects on the
image from physical sensors or the refractive index of the medium of which the light passes.

The challenge that remains after extended research around SISR is how to recover photo-realistic
results with more natural textures and less abnormal extra textures added by the models. The learning
method we will mention in this report and other learning methods that tackle the same problem,
constructs a non linear mapping between an LR image to an HR image H_r = f(L_r, t)
where we have f is the model functional representation that produces a mapping between L_r,
low resolution image and $H_r$ a high resolution image. We have chosen DIV2K data set for this iteration
of the work since it provides variety of images with 2K resolution.
The dataset was collected with inspiration of the 2017 NTIRE challenge data
and study Agustsson 2017 CVPR Workshops.
The challenge was followed by multiple conferences (for example Timofte 2017 CVPR Workshops) and it was 
a precursor for the NTIRE challenge in 2018 with an extended data set. We decided to implement
the SRGAN explained in the paper with tensorflow and keras due to convenience and the presence 
of the DIV2K dataset in tensorflow datasets. The document contains related 
work review and results of the draft model along with issues we faced and how we are handling them. 
The document also contains description of the approach and results.

## Script Instructions
TBA