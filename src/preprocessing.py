import tensorflow as tf

IMAGE_SIZE = (224, 224)


def preprocess_image(image):
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32)
