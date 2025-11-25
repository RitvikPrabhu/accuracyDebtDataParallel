import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

from core.registry import MODELS

@MODELS.register("resnet50_tf")
class ResNet50TFFactory:
    def __init__(self, pretrained: bool = False):
        self.pretrained = pretrained

    def build(self, num_classes: int, input_shape=(224, 224, 3)) -> tf.keras.Model:
        if not self.pretrained:
            return ResNet50(
                include_top=True,
                weights=None,
                classes=num_classes,
                classifier_activation=None,  
                input_shape=input_shape,
            )

        if num_classes == 1000 and input_shape == (224, 224, 3):
            return ResNet50(
                include_top=True,
                weights="imagenet",
                classes=1000,
                classifier_activation=None,  
                input_shape=input_shape,
            )
        base = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
        logits = layers.Dense(num_classes, activation=None, name="fc")(x)  # logits
        model = Model(inputs=base.input, outputs=logits, name="resnet50_tf")
        return model
