import tensorflow as tf
import tensorflow.keras.layers as tfla
import tensorflow.keras.models as tfm
import tensorflow.keras.optimizers as tfo
import tensorflow.keras.losses as tflo
import matplotlib.pyplot as plt

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/Users/zhixinyin/Desktop/Couvolutional Nerual Networks/LeNet-5/mnist_png-master/mnist_png/training",
    image_size = (28, 28),
    color_mode = "grayscale",
    batch_size = 64,
    label_mode = "int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "/Users/zhixinyin/Desktop/Couvolutional Nerual Networks/LeNet-5/mnist_png-master/mnist_png/testing",
    image_size = (28, 28),
    color_mode = "grayscale",
    batch_size = 64,
    label_mode = "int"
)

def preprocess(image, label):
    # white pixels correspond to -0.1 and black correspond to 1.175
    image = tf.cast(image, tf.float32)
    image = -0.005 * image + 1.175

    # pad to 32 * 32
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)

    return(image, label)

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

def activation_function(x):
    return(1.7159 * tf.math.tanh(0.6667 * x))

C3_CONNECTIONS = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5],
    [0, 1, 2, 5], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]
]

class C3Layer(tfla.Layer):
    def __init__(self):
        super().__init__()
        self.partial_convs = []

        for conn in C3_CONNECTIONS:
            conv = tfla.Conv2D(
                filters=1,
                kernel_size=5,
                activation = activation_function,
                padding="valid"
            )
            self.partial_convs.append((conv, conn))

    def call(self, x):
        outputs = []
        
        for conv, conn in self.partial_convs:
            subset = tf.concat([x[:, :, :, i : i + 1] for i in conn], axis = -1)
            outputs.append(conv(subset))
        
        return(tf.concat(outputs, axis=-1))


inputs = tfla.Input(shape=(32, 32, 1))
    
x = tfla.Conv2D(6, kernel_size=5, activation=activation_function, padding="valid",
                        input_shape=(32, 32, 1))(inputs)
    
x = tfla.AveragePooling2D(pool_size = 2)(x)

# customed conv layer with partial connection
x = C3Layer()(x)

x = tfla.AveragePooling2D(pool_size = 2)(x)

# conv layer, but actually act like fully connected layer
x = tfla.Conv2D(120, kernel_size=5, activation=activation_function, padding="valid")(x)

x = tfla.Flatten()(x)

x = tfla.Dense(84, activation = activation_function)(x)

outputs = tfla.Dense(10)(x)

model = tfm.Model(inputs, outputs)

model.compile(
    optimizer = tfo.Adam(learning_rate = 0.001),
    loss = tflo.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds
)

print("\nevaluation:\n")
model.evaluate(test_ds)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()