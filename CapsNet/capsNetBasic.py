#Imports

#Load Data MNIST

#Plot Data (= Show Images)
print ("hi")
#(Set Random seed)

#squash function (epsilon to prevent division by 0!!!)
''' Tensor FLow:
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
'''


#(Conv Layers)

#Primary Capsules

#Digit Capsules

#Routing by Agreement

#Estimated Class Probabilities (=Length of Vectors) What is Direction?

#Margin Loss

#Masking (Mask out "incorrect" Predictions)

#Decoder ("Now let's build the decoder. It's quite simple: two dense (fully connected) ReLU layers followed by a dense output sigmoid layer")

#Reconstruction Loss

#Final Loss (=Reconstruction Loss + Margin Loss)

#Accuracy

#Training 

#Evaluation

#Predictions

#Interpreting the Output Vectors

