import tensorflow as tf 
import os 
from matplotlib import pyplot as plt

class UNet(tf.keras.models.Model):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.conv_block_1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.AveragePooling2D(),
        ])
        
        self.conv_block_2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.AveragePooling2D(),
        ]) 
        
        self.deconv_block_1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
        ])
          
        self.deconv_block_2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
        ])

    def call(self, x):
        x = self.conv_block_1(x)
        skip1 = x 
        
        x = self.conv_block_2(x)
        x = self.deconv_block_1(x)
        
        x = x + skip1
        x = self.deconv_block_2(x)
        
        return x          


def generate():
    # Iterate through images at images/train folder
    # Then, find a mask with exact same name at train folder
    
    for filename in os.listdir("images/train"):
        image_tensor = tf.io.read_file("images/train/" + filename)
        image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
        
        mask_filename = filename.split(".")[0] + ".png"
        
        mask_tensor = tf.io.read_file("masks/" + mask_filename)
        mask_tensor = tf.image.decode_png(mask_tensor, channels=1)
        
        # Change to float and normalize
        
        image_tensor = tf.cast(image_tensor, tf.float32)
        image_tensor = image_tensor / 255.0
        
        mask_tensor = tf.cast(mask_tensor, tf.float32)
        mask_tensor = mask_tensor / 255.0
        
        yield image_tensor, mask_tensor

def fun_stuff():
    for image, mask in generate():
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.show()
        break
    
def main():     
    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=(
            tf.TensorSpec(shape=(480, 640, 3)),
            tf.TensorSpec(shape=(480, 640, 1)),        
        )
    )
    
    dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    for image, mask in dataset:
        print(image.shape)
        print(mask.shape)
        break
    
    model = UNet()
    
    model.compile(optimizer="adam", loss="binary_crossentropy")
    
    model.build(input_shape=(None, 480, 640, 3))
        
    #model.fit(dataset, epochs=1, batch_size=32)

    print(model.summary())

    #model.save("weak_model")

if __name__ == "__main__":
    main()