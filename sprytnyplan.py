import tensorflow as tf
from unet import generate
from matplotlib import pyplot as plt

def main():
    model = tf.keras.models.load_model("weak_model")
    print(model.summary())

    for image, mask in generate():
        image = tf.expand_dims(image, axis=0)

        print(image.shape)

        prediction = model.predict(image)[0]    
        
        #plt.imshow(image[0])
        plt.imshow(prediction)
        
        plt.show()

if __name__ == "__main__":
    main()