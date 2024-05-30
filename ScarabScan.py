from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_directory = 'python/Insectsclasificiation/Photos'

def load_images(image_directory, target_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(image_directory) if not d.startswith('.')])

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(image_directory, class_name)
        image_files = [f for f in os.listdir(class_dir) if not f.startswith('.') and f.lower().endswith(('png', 'jpg', 'jpeg'))]

        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, target_size)
            images.append(image.numpy().astype('float32')) 
            labels.append(class_index)

    return np.array(images, dtype='float32'), np.array(labels), class_names

def normalize_images(images):
    return images / 255.0

def split_data(images, labels, test_size=0.2):
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42)
    return train_images, val_images, train_labels, val_labels

images, labels, class_names = load_images(image_directory)
images = normalize_images(images)
train_images, val_images, train_labels, val_labels = split_data(images, labels)

k = 5  
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []
losses = []

fold_no = 1

def create_model(input_shape, num_classes):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),  
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),  
        Dense(num_classes, activation='softmax')
    ])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


for train_index, val_index in skf.split(train_images, train_labels):
    train_fold_images, val_fold_images = train_images[train_index], train_images[val_index]
    train_fold_labels, val_fold_labels = train_labels[train_index], train_labels[val_index]

    
    model = create_model((128, 128, 3), 3)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_generator = datagen.flow(train_fold_images, train_fold_labels, batch_size=32)

    model.fit(
        train_generator,
        steps_per_epoch=len(train_fold_images) // 32,
        epochs=500,
        validation_data=(val_fold_images, val_fold_labels)
    )

    val_loss, val_accuracy = model.evaluate(val_fold_images, val_fold_labels, verbose=0)
    accuracies.append(val_accuracy)
    losses.append(val_loss)

    print(f"Fold {fold_no}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    fold_no += 1

print(f"Average Validation Accuracy: {np.mean(accuracies)}")
print(f"Average Validation Loss: {np.mean(losses)}")