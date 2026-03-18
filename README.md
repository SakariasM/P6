HUMAN DETECTION MODEL ON EDGE DEVICES (raspberry pi 5)

```
P6/
├── src/
│   ├── teacher/              # Phase 1: Extract teacher knowledge
│   │   ├── __init__.py
│   │   ├── predictions.py           # Basic YOLO inference + predictions
│   │   ├── feature_extractor.py     # Extract intermediate features
│   │   └── hybrid_predictions.py    # Combine predictions + features
│   │
│   ├── student/              # Phase 2: Student model architecture
│   │   ├── __init__.py
│   │   └── student_model.py         # Lightweight model + feature adapters
│   │
│   ├── training/             # Phase 3: Train student
│   │   ├── __init__.py
│   │   └── hybrid_distillation_train.py  # Dual-loss training pipeline
│   │
│   ├── utils/                # Helper utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Dataset loaders
│   │   └── download_dataset.py      # Dataset downloading
│   │
│   └── legacy/               # Old code (archived)
│       ├── __init__.py
│       ├── main.py
│       └── tester.py
│
├── data/                     # Training data
│   └── images/               # Input images
│
├── results/                  # Output predictions
│   └── hybrid_predictions/
│       ├── hybrid_teacher_predictions.pt   # Teacher knowledge
│       └── metadata.json                   # Dataset info
│
├── trained_models/           # Trained student models
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
│
├── yolo26n.pt               # Teacher model weights

```


usage:

python -m venv .

source .venv/bin/activate


# For pulling attention class predictions
python3 predictions.py --model yolo26n-seg.pt --input ../test_data/sample/ --output ../results --format pickle


# Run the complete pipeline
./run_hybrid_distillation.sh

or manually

# Step 1: Extract teacher predictions + features
python src/hybrid_predictions.py \
    --model yolo26n.pt \
    --input ./data/images \
    --output ./hybrid_predictions \
    --batch-size 8 \
    --person-only



# For COCO training set (118K images)
python src/download_dataset.py --dataset coco --split train2017 --output data

# For Open Images (10K images)
python src/download_dataset.py --dataset open-images --num-images 10000 --output data

# For custom URLs
python src/download_dataset.py --dataset custom --url-file image_urls.txt --output data

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

```
             *     ,MMM8&&&.            *
                  MMMM88&&&&&    .
                 MMMM88&&&&&&&
     *           MMM88&&&&&&&&
                 MMM88&&&&&&&&
                 'MMM88&&&&&&'
                   'MMM8&&&'      *
           /\/|_      __/\\
          /    -\    /-   ~\  .              '
          \    = Y =T_ =   /
           )==*(`     `) ~ \
          /     \     /     \
          |     |     ) ~   (
         /       \   /     ~ \
         \       /   \~     ~/
  jgs_/\_/\__  _/_/\_/\__~__/_/\_/\_/\_/\_/\_
  |  |  |  | ) ) |  |  | ((  |  |  |  |  |  |
  |  |  |  |( (  |  |  |  \\ |  |  |  |  |  |
  |  |  |  | )_) |  |  |  |))|  |  |  |  |  |
  |  |  |  |  |  |  |  |  (/ |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
```
