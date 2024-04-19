import tensorflow as tf
import tensorflow_federated as tff

# Load a sample dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Preprocess the dataset
def preprocess(dataset):
    def batch_format_fn(element):
        return (tf.expand_dims(element['pixels'], -1), tf.one_hot(element['label'], 10))

    return dataset.map(batch_format_fn)

# Define a simple model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Wrap the Keras model with a TFF model
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(keras_model, input_spec=emnist_train.element_spec, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Define federated averaging process
fed_avg = tff.learning.build_federated_averaging_process(model_fn)

# Initialize the server state
state = fed_avg.initialize()

# Run federated training
for _ in range(5):
    state, metrics = fed_avg.next(state, [emnist_train.take(100)])
    print(f'Training metrics: {metrics}')

# Evaluate federated model on test dataset
evaluation = tff.learning.build_federated_evaluation(model_fn)
test_metrics = evaluation(state.model, [emnist_test.take(100)])
print(f'Test metrics: {test_metrics}')
