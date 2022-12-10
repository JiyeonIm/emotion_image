# Web-application-sentimental-heatmap
Web application for image emotion detection with sentimental heatmaps. 

![home](https://user-images.githubusercontent.com/63252403/206867019-b2e41523-eab4-4bef-b571-d505650aeeae.png)

## Overview🧐
This is a simple web application for SKKU's course "Data-Driven Service Design and AI Application(2022-2)". This application employs image emotion detection with sentimental heatmaps to better understand the model's decision. This can be used by a company interested in monitoring people's opinions or by users who want to understand other people's emotions through images.


This repository includes implementation code for ✅  [model training](https://github.com/JiyeonIm/emotion_image/tree/main/model) ✅  [web application with flask](https://github.com/JiyeonIm/emotion_image)

Examples are as follows:
![예시](https://user-images.githubusercontent.com/63252403/206866970-c1b2537e-d7f1-45a2-ae0a-28004533f757.png)


## Requirements
For experimental setup, requirements.txt lists down the requirements for running the code on the repository. Note that a cuda device is required for model training. (It is not required for web application/inference). The requirements can be downloaded using,
  ```
  pip install -r requirements.txt
  ```

## Usage
1. Clone the repository.
  ```
  git clone https://github.com/JiyeonIm/emotion_image.git
  ```
  
2. Train the model. You can train the model as follows:
  ```
  python train.py
  ```
  Please note that training data should be located at [data](https://github.com/JiyeonIm/emotion_image/tree/main/model/data) directory. Here, we used [FI dataset](https://ojs.aaai.org/index.php/AAAI/article/view/9987) which consists of images labeled with corresponding emotion categories.


3. The checkpoints of the best validation performance will be saved in [checkpoints](https://github.com/JiyeonIm/emotion_image/tree/main/checkpoints) directory. You can further train model or use for inference with this checkpoint.

4. Run the application as follows:
  ```
  python app.py
  ```
Open a browser and visit http://localhost:5000, and see what you get!
