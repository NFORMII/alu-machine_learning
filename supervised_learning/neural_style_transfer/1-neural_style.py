def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(vgg.input, outputs)