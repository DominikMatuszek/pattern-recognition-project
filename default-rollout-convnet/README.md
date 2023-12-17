# Masked image saliency map prediction

### Approach 3: Basic attention rollout and convnet student

*Code availabe [here](https://github.com/kacpermarzol/Pattern-recognition-project-basic-rollout-convnet")*

We want to try a convolutional nerual network as a student for a visual transformer teacher. The plan was to downsample
the image for studnet model and make it learn the attention from the teacher model (which uses the full image):

![pattern](images/pattern.jpg)

For all experiments we used ResNet18 (11 million parameters) with trained weights. In the first attempt we used deit
tiny as a teacher, which has 5 million parameters, we also used exactly the same pictures as input for teacher and
student. The results were promising:

![readme1](images/readme1.png)
![readme2](images/readme2.png)

We also checked that the model recognizes the same objects, which are in different places at on the picture:

![readmex](images/readmex.png)

Then we wanted to use a bigger architecture as a student, so we used deit large, altough some of its attention rollouts
are not very clear for humans:
![readme3](images/readme3.png)

This time we wanted to use the downsizing approach for student, but to still be able to use the pretrained ResNet, we
came up with an idea to resize the image to a small resolution and then resize it to the first size, so that the quality
of the picture is worse, while still being a 224x224 image. Here is an example of a original 224x224 picture, and a
picture that has been resized to 70x70, and then again upsampled to size 224x224:

![readme5](images/readme4.png)
![readme5](images/readme5.png)

Additionaly we added some heavy augmenations. Unfortunately this approach didn't manage to learn anything, moreover
after a few epochs it blew up and started outputing nans for everything.

We reduced the augmentations, the model didn't blow up, yet still it didn't manage to learn anything meaningful:

![readme6](images/readme6.png)
![readme7](images/readme7.png)

Changing the learning rate helped and we managed to get some nice results:

![readme8](images/readme8.png)
![readme9](images/readme9.png)

The loss seemed to be going down on the training set:
![readme10](images/readme10.png)

But it fails the different object placement test:
![readmez](images/readmez.png)

It might be due to the vague outputs from the attention rollout, here the model seems to be working well (for human
standards), while the attention rollout from the teacher is meaningless:

![readme11](images/readme11.png)
