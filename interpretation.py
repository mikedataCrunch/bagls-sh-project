from tensorflow.keras.callbacks import Callback
import tensorflow as tf

import numpy as np
import wandb
import cv2

VAL_TABLE_NAME = "val-predictions" 
GRADCAM_TABLE_NAME = "gradcam-results"

class GradCAM:
    """
    Reference:
        https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
        https://github.com/ayulockin/interpretabilitycnn/
    """

    def __init__(self, model, layerName):
        self.model = model
        self.layerName = layerName

        self.gradModel = tf.keras.models.Model(inputs=[self.model.inputs],
                                               outputs=[self.model.get_layer(self.layerName).output, self.model.output])

    def compute_heatmap(self, image, classIdx, eps=1e-8):
        with tf.GradientTape() as tape:
            tape.watch(self.gradModel.get_layer(self.layerName).variables)
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)

            if len(predictions) == 1:
                # Binary Classification
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap


    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_HOT):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

    
class GRADCamLogger(Callback):
    def __init__(self, generator, layer_name, num_log_batches=1):
        super(GRADCamLogger, self).__init__()
        self.generator = generator
        self.num_log_batches = 1
        self.layer_name = layer_name
        self.flat_class_names = [k for k, v in generator.class_indices.items()]


    def on_epoch_end(self, logs, epoch):
        val_data, val_labels = zip(
             *(self.generator[i] for i in range(self.num_log_batches))
         )
        val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)
        
        
        images, gradcams = [], []
        true_labels, pred_labels = [], []
        pred_scores = []
        
        # init gradcam class
        cam = GradCAM(self.model, self.layer_name)
  
        for image, label_arr in zip(val_data, val_labels):
            label_index = np.argmax(label_arr)
            # get and append true label
            true_label = self.flat_class_names[label_index]
            true_labels.append(true_label)
            
            image = np.expand_dims(image, 0)
            pred = self.model.predict(image)
            
            # get and append prediction score, class index, class name
            pred_score = np.max(pred[0])
            # get prediction index for gradcam computation
            pred_index = np.argmax(pred[0])
            pred_label = self.flat_class_names[pred_index]
            pred_scores.append(pred_score)
            pred_labels.append(pred_label)
            

            
            # get gradcam heatmap, use pred_index
            heatmap = cam.compute_heatmap(image, pred_index)
            
            image = image.reshape(image.shape[1:])
            image = image * 255
            image = image.astype(np.uint8)
    
            # overlay gradcam heatmap results to image
            heatmap = cv2.resize(heatmap, (image.shape[0],image.shape[1]))
            (heatmap, output) = cam.overlay_heatmap(heatmap, image, alpha=0.5)
            
            # append original and gradcam overlay image
            images.append(image)
            gradcams.append(output)
            
        
        # log validation predictions alongside the run
        columns = ["image", "gradcam-overlay", "pred_score", "pred", "truth"]
        gradcam_table = wandb.Table(columns = columns)
        
        for item in zip(images, gradcams, pred_scores, pred_labels, true_labels):
            row = [wandb.Image(item[0]), wandb.Image(item[1]), *item[2:]]
            gradcam_table.add_data(*row)
                                  
        wandb.run.log({GRADCAM_TABLE_NAME : gradcam_table})


class ValLog(Callback):
    """Custom callback to log validation images at the end of each training epoch.
    
    Reference: https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA
    """
    def __init__(self, generator=None, num_log_batches=1):
        self.generator = generator
        self.num_batches = num_log_batches
        
        # store full names of classes
        self.flat_class_names = [k for k, v in generator.class_indices.items()]

    def on_epoch_end(self, epoch, logs={}):
        # collect validation data and ground truth labels from generator
        val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
        val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)
    
        # use the trained model to generate predictions for the given number
        # of validation data batches (num_batches)
        val_preds = self.model.predict(val_data)
        true_ids = val_labels.argmax(axis=1)
        max_preds = val_preds.argmax(axis=1)
    
        # log validation predictions alongside the run
        columns=["id", "image", "pred", "truth"]
        for a in self.flat_class_names:
            columns.append("score_" + a)
        predictions_table = wandb.Table(columns = columns)
        
        # log image, predicted and actual labels, and all scores
        for filepath, img, top_guess, scores, truth in zip(self.generator.filenames,
                                                           val_data, 
                                                           max_preds, 
                                                           val_preds,
                                                           true_ids):
            img_id = filepath.split('/')[-1].split(".")[0]
            row = [img_id, wandb.Image(img), 
                 self.flat_class_names[top_guess], self.flat_class_names[truth]]
            for s in scores.tolist():
                row.append(np.round(s, 4))
            predictions_table.add_data(*row)
            
        wandb.run.log({VAL_TABLE_NAME : predictions_table})
        
        
 