HUMAN DETECTION MODEL ON EDGE DEVICES (raspberry pi 5)


The goal of this project is to distill a model from yolo26n which is optimized for detecting humans on low powered hardware such as a raspberry pi 5.


The overall plan:

Goal: 70% mAP & minimun 24 fps on RPi5.

1. create a program which can save the teacher inference results from yolo26n (predictions.py). 

2. Benchmark teacher model, 


3. find a suitable dataset for training and validation. Good performance: 20,000-50,000 images.

4. Create a custom lightweight cnn student model that can distill from the results of   yolo26n using response-level distillation. (also design distillation loss, KL divergence + hard label loss)

4.5. Test quantized model for speed vs accuracy?

5. Keep optimizing the training process to improve accuracy and efficiency of the model.

6. Implement inference engine in c++ (some already exists).

