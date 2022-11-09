# necessary to work with machine learning
import tensorflow as tf

# preparing data for model
from tensorflow.keras.applications.vgg19 import preprocess_input

# creating models form input and output list with training features
from tensorflow.keras.models import Model

# for working with charts and files
import matplotlib.pyplot as plt pp

# necessary for vectors
import numpy as np 

# load image from filex
def load(img): 
  
  # open file
  img = plt.imread(img); 
  
  # convert pixel values to float32
  img = tf.image.convert_image_dtype(img, tf.float32) 
  
  # compress to 400x400
  img = tf.image.resize(img, [400, 400]) 
  
   # add one outer dimention
  img = img[tf.newaxis, :]
  return img

# load both iamges
content = load("content.jpg")
style = load("style.jpg")

# load vgg19 without top layer for detection
vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg19.trainable = False 

#define layers that we are going to derive our features from
contentLayers = ['block1_conv2']
styleLayers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

#define length of the upper mentioned
numContentLayers = len(contentLayers)
numStyleLayers = len(styleLayers)

# craete new model for wanted outputs
def mini_model(layer_names, model):

  # obtain outputs for wanted layers
  outputs = [model.get_layer(name).output for name in layer_names]

  # create new model for desired outputs using vgg input data
  model = Model([vgg19.input], outputs)
  return model

#calculate gram matrix
def gram_matrix(tensor):
  
  temp = tensor
  # remove the outer dimention
  temp = tf.squeeze(temp) 
  
  # reshape the image to 2D
  temp = tf.reshape(temp, [temp.shape[2], temp.shape[0]*temp.shape[1]]) 
  
  # multiply the initial matrix by transposed version of itself 
  result = tf.matmul(temp, temp, transpose_b=True) 
  
  # bring back 1 dimention
  gram = tf.expand_dims(result, axis=0) 
  
  return gram # return the gram matrix


class NST(tf.keras.models.Model):
  def __init__(self, styleLayers, contentLayers):
    super(NST, self).__init__()
    self.vgg19 =  mini_model(styleLayers + contentLayers, vgg19)
    self.styleLayers = styleLayers
    self.contentLayers = contentLayers
    self.NumStyleLayers = len(styleLayers)
    self.vgg19.trainable = False

  def call(self, inputs):
    
    # denormalize pixel values
    inputs = inputs*255.0
    
    # Preprocess to suit VGG19 stats
    processedInput = preprocess_input(inputs)
    
    # obtain the output from the mini network
    outputs = self.vgg19(processedInput)
    
    # separate style and content 
    styleOutputs, contentOutputs = (outputs[:self.NumStyleLayers],
                                      outputs[self.NumStyleLayers:])

    # calculate the gram matrices for each style layer
    styleOutputs = [gram_matrix(style_output)
                     for style_output in styleOutputs]

    # create a content dictionary of layer name: output
    contentDict = {names:value
                    for names, value
                    in zip(self.content_layers, contentOutputs)}

    # create a style dictionary of layer name: output
    styleDict = {names:value
                  for names, value
                  in zip(self.style_layers, styleOutputs)}

    # return both dictionaries
    return {'content':contentDict, 'style':styleDict}

# initialize the model
extractor = NST(styleLayers, contentLayers)

# extract style target data
styleTargets = extractor(style)['style']

# extract content target data
contentTargets = extractor(content)['content']

 # Custom weights for style and content updates
styleWeight = 100
contentWeight = 100

# Custom weights for different style layers
styleWeights = {'block1_conv1': 1.0, 'block2_conv1': 1.0,'block3_conv1': 1.0,'block4_conv1': 1.0,'block5_conv1': 1.0,}

# loss function for optimization
def totalLoss(outputs):
  
    #take both style and content from the output dictionary
    styleOut = outputs['style']
    contentOut = outputs['content']
    
    #calculate style loss using the formula
    styleLoss = tf.add_n([styleWeights[name]*tf.reduce_mean((styleOut[name]-styleTargets[name])**2)
                          for name in styleOut.keys()]) * styleWeight / numStyleLayers
    
    #calculate content loss using the formula
    contentLoss = tf.add_n([tf.reduce_mean((contentOut[name]-contentTargets[name])**2)
                            for name in contentOut.keys()]) * contentWeight / numContentLayers
    
    result = styleLoss + contentLoss
    return result
  
# necessary for calculation on explicit regularization term
def highPassXY(img):
  xVar = img[:, :, 1:, :] - img[:, :, :-1, :]
  yVar = img[:, 1:, :, :] - img[:, :-1, :, :]

  return xVar, yVar

# for regularization so that more 
def totalVariationLoss(img):
  xDel, yDel = highPassXY(img)
  return tf.reduce_sum(tf.abs(xDel)) + tf.reduce_sum(tf.abs(yDel))
  
# weight for total variation
totalVariationWeight=30  

# select adam optimizer with learning rate of X
optimizer = tf.optimizers.Adam(learning_rate=0.02)

# define the training function
@tf.function()
def trainStep(img):
  with tf.GradientTape() as tape:
    
    #obtain the outputs of the network for an image
    outputs = extractor(img)
    
    # Calculate the loss for that image
    loss = totalLoss(outputs)
    
    # calculate variation loss 
    loss += totalVariationWeight * totalVariationLoss(img)
  
  # calculate the gradient of the image and the loss results
  grad = tape.gradient(loss, img)
  
  # apply the gradient on the image
  optimizer.apply_gradients([(grad, img)])
  
  # clip the pixel values to [0,1]
  img.assign(tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0))

# define randomized image
target = tf.Variable(content)

#define number of epochs and the number of steps
epochs = 2
epochSteps = 100

#current step
step = 0

# for each epoch
for n in range(epochs):
  
  # for each steps 
  for m in range(epochSteps):
    step += 1
    
    # train the network
    trainStep(target)
    print("epoch step:", step)
    
# display the image after each epoch
plt.imshow(np.squeeze(target.read_value(), 0))
plt.title("Train step: {}".format(step))
plt.show()