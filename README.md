# NeuralStyleTransfer
My implementation of the Neural Style Transfer Algorithm for the assignment at my university for the topic of Biologically Inspired Artificial Intelligence.

Fine art is the domain of human behaviour in which people have mastered the skill of obtaining unique visual media. 
Thus far, we have no (direct) algoritmic solution of quality equal to humans.
But the face-recognition and object-recognition software has been yet provided.
Now, we have recently come up with artificial neural network system that aims to generate artistic images.
The system breaks existing media to ‘content’ and ‘style’ in neural representations that can be recombined to create new images.
This method provides a path to understand how humans create and perceive artistic imagery.

The idea for style transfer have been there for so long such as since 1980s.
Later in 2000s people have come up with first algorithms but only since 2015 it have been used using neural networks.
People wanted to solve this problem for centuries since the 80s and the filters and stylizing techniques so that later they could use patch-based methods such as IMAGE ANALOGIE.
The current solution have been known only when Leon Gatys wrote a paper with his collegues called An Algorithm for Neural Style. In this paper, they have came up with the fact that you could take out the content and the style of an image after you feed it to convolutional neural network.

CNNs consist of layers of computional units that process image information in feed forward manner. Each layer can be thought of as image filters and each of them extracts a certain feature from the input. 
The input of a certain layer is what we call FEATURE MAPS - the ones that we mentioned the layer has extracted. 
CNN trained for Object Recognition develop increasingly explicit feature maps the more you go deep into the network. Meaning that more processed layers are more concerned about the high level content of an image, rather than details.
We can very easily visualize the feature maps in a certain layer and see what does it contain by reconstruction.
Higher layers capture higher level features such as entire objects or entire faces.
Lower layers capture lower level features such as exact pixel values.
We call higher level features as content representation.

A Neural Network that have been trained on ImageNet (public google dataset, containing more than 14kk hand annotated images and over 20k categories) to search for objects and localizations on images. It participated in a challenge of Large Scale Visual Recognition Challenge. Did not win the contest.
It can help in finding the semantics of an image.
We use it as such:
we take our precious images, we feed them to VGG19, we take out feature maps from the layer, and these feature maps are our input image content
In the paper they have used feature space provided by 16 conv and 5 pooling layers of the 19 layer VGG-network. 

An input image is treated as a set of filtered images at each layer.  The number of filters increases the more we go into the network (to the right).
The size of the filtered images is reduced due to max-pooling, which reduces the total number of units per layer. 
We can visualize each processing stage by reconstructing the image from the layers output. We reconstruct input layers from “convX_1” from the original VGG.
Reconstruction from first three layers is perfect. In higher layers the details are lost while the higher level information is preserved. 
On top of the image we build a new feature space that should capture the styles.
We can build the style representations based on many combinations of layers, such as conv1_1 up to conv5_1.
This way, we can derive what exactly makes up the style of the pictures, the more layers the more you care about style and less about the content. The style representation consists of the multi-scale feature maps taken from many layers.

We first load the images and convert them to vectors to feed to VGG (we prepare a special function for that loading).
We first copy our VGG19 network. Then, we declare the layers from which we are going to derive feature maps. 
We set up a “mini model” by collecting VGG19 outputs at arbitralny layers, and then we feed it to the new model giving it vgg19 input data as an input and the desired output features as output.

We obtain output image by rewriting it as an optimization problem. We can write a function of similarity between the content and the new noise image in the current step. 
We can do the same with the style of the image. But to do so, we need to separate the content and the style from the base images, which is very non-trivial thing to do.
Each layer defined filter bank whose complexity increases the deeper you go.
Input image of vector x is encoded in each layer by the filter. 
Layer has N filters, therefore it has N feature maps of size M.
To visualize this encoded information we perform gradient descent on a white noise image to find an image that matches the derived feature data. 
Vector p and vector x can be the original image and the generated image. 
P^l and F^l will be their feature representations at layer l.
We can define the squared error loss between the two.

We calculate the gram matrix between style feature maps to obtain the correlations between them, so we can merge them together in one matrix.
Let’s say we have a four feature map matrix of 20x20x16.  
We first flatten them out, so we have 400x16 so 16 matrices of 400 pixels.
Then we transpose the copy of this 400x16 matrix and multiply it by that. Resulting in 4x4 matrix of correlations in feature maps, where each value denotes similarities between feature maps.
We then use this information to calculate the style loss, considering the gram matrix of each of the layers in the CNN.

On top of the image we build a new feature space that should capture the styles.
We can build the style representations based on many combinations of layers, such as conv1_1 and conv2_1 and/or conv3_1 up to conv5_1.
This way, we can derive what exactly makes up the style of the pictures, the more layers the more you care about style and less about the content.
The style representation consists of the multi-scale feature maps taken from many layers.
We then try to minimize the Gram matrix from the original image and the Gram matrix of the image to be generated.

We construct total loss function and use it for gradient descent, which is necessary to recalculate weights and biases in the network in each step.
The alpha and beta constants are here to denote the intensity of both content and style.

The larger image, the more time and memory you need to “allocate”, for the network to compute. The GPU and CUDA technology did not allow me to use images in FullHD, caused errors that are going to be mentioned further, so I had to limit myself to safe image size of 400x400.

The more epochs and steps you assign, the more stylized and abstract the image you are going to obtain, which is quite obvious, because it is the number of times you run the loss function and provide it to the gradient descent “tweaking” the image a little.
Most of the tests were done on the 10 epochs 100 steps. Maybe if I used more epochs, then the effects would be more optimistic.

During my development process I have found out that doing calculation on GPU makes the entire process faster exponentially.
But it was tricky for me to install the optimal versions of graphics drivers and remove the newest ones since tensorflow development does not follow driver development instantly. It also required some tweaking in system32 folders, changing the PATH variable and even moving files by hand since the installation does not put everything necessary in the right places. 
It really took some time to set it up, but it was worth it. Before I set up CUDA each step took up to 10 seconds to finish, while after CUDA it got almost instant which made me able to run the gradient descent over the image in way more epochs that I used to in reasonable time.

Changing the size of the images obviously increases the number of time the algorithm spends on the execution. But that wasn’t the only issue for me, I believe that due to limited VRAM on my graphics card I am not able to run the algorithm over the fullHD images. 
For this project I have used NVIDIA GTX 1050 Ti which has only 4GB of VRAM, and despite that, the tensorflow warned that it might be too little. 
Even when I run the program over the compressed images of size 400x400, then I still find allocator errors that requires me to free more VRAM (There were no such issues with CPU).
The program then crashes when it tries to do some calculation over the tensor. But I haven’t been able to tell whether that is the case.

The NST is a very promising technology, but it might require more experiments to do it right. Every image requires some tweaking before you learn how to establish optimal parameters so that it looks good.
I have also found out that the algorithm is more effective when given an image resembling texture, rather than image. I have also found out that the parameters that matter the most are learning rate and the style weights.

Full presentation with examples for the given parameters:
https://docs.google.com/presentation/d/1M5cWEs8XVTIjdfhi8BjsWM0x0yVC7KTQUon9An-NmQA/edit?usp=sharing

Sources:

A Neural Algorithm of Artistic Style
https://arxiv.org/abs/1508.06576
Advanced Theory on NST
https://www.youtube.com/watch?v=XWMwdkaLFsI
How Deep Dreams (Basically) Work
https://www.youtube.com/watch?v=xkWgwxKxWAE
Two minute papers 
https://www.youtube.com/watch?v=Uxax5EKg0zA

