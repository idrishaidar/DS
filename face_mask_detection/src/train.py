from dataset import ImageDataset
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from config import images_path, annot_path

# define the model with Xception
def train_xception(train_dataset, test_dataset):
    # n_classes = len(np.unique(train_labels))
    n_classes = 4
    base_model = Xception(
        weights='imagenet', include_top=False
    )
    inputs = Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    base_model = base_model(x)
    avg_pool = GlobalAveragePooling2D()(
        base_model)
    class_output = Dense(n_classes, activation='softmax', name='class_output')(avg_pool)
    loc_output = Dense(4, name='loc_output')(avg_pool)
    model = Model(
        inputs=inputs,
        outputs=[class_output, loc_output]
    )

    # compile the model
    model.compile(
        loss=['sparse_categorical_crossentropy', 'mae'],
        # loss_weights=[0.5, 0.5],
        optimizer='sgd', metrics=['accuracy', 'mae']
    )

    # create batch dataset
    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset_batched = train_dataset.shuffle(
        SHUFFLE_BUFFER_SIZE
    ).batch(BATCH_SIZE)

    test_dataset_batched = test_dataset.batch(BATCH_SIZE)

    # model training
    model.fit(
        train_dataset_batched, epochs=5,
        validation_data=test_dataset_batched)

    return model

def run():
    # images_path = '../dataset/images/'
    # annot_path = '../dataset/annotations/'

    face_mask_dataset = ImageDataset(images_path, annot_path)
    train_dataset, test_dataset = face_mask_dataset.create_dataset_tensors()

    model = train_xception(train_dataset, test_dataset)
    return model

if __name__ == '__main__':
    run()