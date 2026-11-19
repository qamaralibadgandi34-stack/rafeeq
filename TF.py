import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model=keras.Sequential([
    layers.flatten(input_shape=(28,28)),
    layers.Dense(512,activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10,activation="softmax")])
log_dir="logs/fit/"+datetime.now().strftime("%y%m%d%-%H%M%S")
tensorboard_callbacks=keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
model.compile(
    optimizer=keras.optimizers.SGD(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model.summary()
model.fit(
    x_train,y_train,
    epochs=10,
    validation_data=(x_test,y_test),
    callbacks=[tensorboard_callbacks])
test_loss,test_accuracy=model.evaluate(x_test,y_test)
print("Test Loss:",test_loss)
print("Test Accuracy:",test_accuracy)