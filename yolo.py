"""imports"""
import os
import shutil
import numpy as np
from ultralytics import YOLO

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
        """ Converts the data file to YOLO format and write out a file."""
        with open(self.input_file, 'r') as file:
            for line in file:
                line = line.strip()
                data = line.split(',')
                patient_id = int(data[0])
                x_min, y_min, x_max, y_max = int(data[1]), int(data[2]), int(data[3]), int(data[4])

                # pixel width and height
                width = 50
                height = 50
                # Calculate annotations
                x_center = (x_min + x_max)/ (2 * width)
                y_center = (y_min + y_max)/ (2 * height)
                width = (x_max - x_min) / width
                height = (y_max - y_min) / height
                # Check if output directory exist
                os.makedirs(self.output_dir, exist_ok=True)
                output_file = os.path.join(self.output_dir, '{}.txt'.format(patient_id))
                # Write the output file to correspond the image
                with open(output_file, 'wb') as out_file:
                    out_file.write(f"1 {x_center} {y_center} {width} {height}\n")

    def configurate_dir(self, img_dir):
        """
        Configures directory with image and label train and test data
        :input; img_dir: image directory
        """
        # Make directories
        os.makedirs(f"{self.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)

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
            shutil.move(f"{img_dir}/{t_file}", f"{self.output_dir}/images/train/{t_file}")
            shutil.move(f"{self.output_dir}/{idx}.txt", f"{self.output_dir}/labels/train/{idx}.txt")

        # Move files to the validation dir
        for v_file in val_files:
            idx = v_file.split('.')[0]
            shutil.move(f"{img_dir}/{v_file}", f"{self.output_dir}/images/val/{v_file}")
            shutil.move(f"{self.output_dir}/{idx}.txt", f"{self.output_dir}/labels/val/{idx}.txt")


    def train_model(self):
        """ Train the model with YOLO"""
        model = YOLO("yolov8n.pt")
        model.train(data='data_sets.yaml', epochs= 10, single_cls = True)

    def predict(self):
        """ Predict the location of a tumor on an unknown data """
        pass

if __name__ == '__main__':

    IMG = '../data/images'
    INPUT = '../data/coords-idc.txt'
    OUTPUT = '../data/output'
    yolo = YOLO_Network(INPUT, OUTPUT)
    #yolo.convert_to_yolo()
    #yolo.configurate_dir(IMG)
    yolo.train_model()
    #yolo.predict()
