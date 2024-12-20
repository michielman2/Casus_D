"""imports"""
import os
import shutil
import numpy as np
from ultralytics import YOLO
from PIL import Image

class YOLO_Network:
    """ class YOLO Network"""
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

    def __str__(self):
        """ String representation of YOLO object. """
        return (f" YOLO Netwerk class;\n"
                f"input file: {self.input_file},\n"
                f"output directory: {self.output_dir})")

    def __repr__(self):
        """ Technical representation of YOLO object. """
        pass



    def convert_to_yolo(self):
      """ Converts the data file to YOLO format and writes out a file. """
      with open(self.input_file, 'r') as file:
         for line in file:
               line = line.strip()
               data = line.split(',')
               patient_id = int(data[0])
               x_min, y_min, x_max, y_max = map(int, data[1:5])

               # Get image dimensions dynamically
               image_path = os.path.join('D:\Rene\Documents\school\kanker_modeleren\casus4\datasets\images', f'{patient_id}.jpeg')  
               if not os.path.exists(image_path):
                  print(f"Image not found for Patient ID {patient_id}: {image_path}")
                  continue

               with Image.open(image_path) as img:
                  image_width, image_height = img.size

               # Calculate normalized annotations
               x_center = ((x_min + x_max) / 2) / image_width
               y_center = ((y_min + y_max) / 2) / image_height
               bbox_width = (x_max - x_min) / image_width
               bbox_height = (y_max - y_min) / image_height

               # Debugging annotations
               print(f"Processing - Patient ID: {patient_id}, x_center: {x_center}, y_center: {y_center}, "
                     f"bbox_width: {bbox_width}, bbox_height: {bbox_height}")

               # Ensure all values are within [0, 1]
               if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                  print(f"Skipping invalid annotation: {line}")
                  continue

               # Check if output directory exists
               os.makedirs(self.output_dir, exist_ok=True)
               output_file = os.path.join(self.output_dir, f'{patient_id}.txt')

               # Write the output file
               with open(output_file, 'w') as out_file:
                  out_file.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


    def configurate_dir(self, img_dir):
        """
        Configures directory with image and label train and test data
        :input; img_dir: image directory
        """
        # Make directories
        os.makedirs(f"{self.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/val", exist_ok=True)

        # Initialize the image files and shuffle
        image_files = []
        for img in os.listdir(img_dir):
            if img.endswith('.jpeg'):
                image_files.append(img)
        np.random.shuffle(image_files)


        # Split the train and val files
        train_ratio = 0.75 # ratio of train to 35 val ratio
        split = int(train_ratio * len(image_files))
        train_files = image_files[:split]
        val_files = image_files[split:]

        # Move files to the train dir
        for t_file in train_files:
            idx = t_file.split('.')[0]
            shutil.copy(f"{img_dir}/{t_file}", f"{self.output_dir}/images/train/{t_file}")
            shutil.copy(f"{self.output_dir}/{idx}.txt", f"{self.output_dir}/labels/train/{idx}.txt")

        # Move files to the validation dir
        for v_file in val_files:
            idx = v_file.split('.')[0]
            shutil.copy(f"{img_dir}/{v_file}", f"{self.output_dir}/images/val/{v_file}")
            shutil.copy(f"{self.output_dir}/{idx}.txt", f"{self.output_dir}/labels/val/{idx}.txt")


    def train_model(self):
         """ Train the model with YOLO"""
         model = YOLO("yolov8n.pt")
         model.train(data=r'D:\Rene\Documents\school\kanker_modeleren\casus4\datasets\data_sets.yaml', epochs=10, single_cls=True)

    def predict(self, input_images_dir, output_predictions_dir):
         """
         Predict the location of a tumor on unknown data using the trained YOLO model.
         :param input_images_dir: Directory containing images for prediction.
         :param output_predictions_dir: Directory to save prediction results.
         """
         # Ensure output directory exists
         os.makedirs(output_predictions_dir, exist_ok=True)

         # Load the trained YOLO model
         model = YOLO("runs/detect/train13/weights/best.pt")  # Replace with the path to your trained model

         # # Iterate through all images in the input directory
         # for img_file in os.listdir(input_images_dir):
         #    if img_file.endswith(('.jpeg', '.jpg', '.png')):
         #          img_path = os.path.join(input_images_dir, img_file)
                  
         #          # Perform prediction
         #          results = model.predict(source=img_path, save=False, save_txt=False)

         #          # Parse and save predictions
         #          output_file = os.path.join(output_predictions_dir, f"{os.path.splitext(img_file)[0]}.txt")
         #          with open(output_file, 'w') as out_file:
         #             for result in results[0].boxes:
         #                cls, x_center, y_center, bbox_width, bbox_height, confidence = result.tolist()
         #                out_file.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {confidence:.6f}\n")

         #          print(f"Predictions saved for {img_file} at {output_file}")
         model.predict("datasets/output/images/val", save=True)


if __name__ == '__main__':

    IMG = 'datasets/images'
    INPUT = 'coords-idc.txt'
    OUTPUT = 'datasets/output'
    yolo = YOLO_Network(INPUT, OUTPUT)
    yolo.convert_to_yolo()
    yolo.configurate_dir(IMG)
   #  yolo.train_model()
   #  yolo.predict("predict_input", "predict_output")
