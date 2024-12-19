# TactifAI

TactifAI is a cutting edge computer vision and AI engine focused on match and player analysis.

## Authors

* Michael Baart - [@LinkedIn](https://www.linkedin.com/in/michael-baart/) - [@GitHub](https://github.com/mbaart)

## Roadmap

### Approach One - Spatio-Temporal Pose Estimation GCN

Detection, to get ROI -> Segmentation, to get a mask -> Pose Estimation -> Insights

* Computer Vision
  * Object Detection, Segmentation, Key Point Detection, Pose Estimation

* Model Training - Human Pose and Action Recognition
  * Milti-object 3D pose traking over short windows of time?

* Key Point detection - Non-players
  * Ball
  * Reff and linesmen
  * Goalies vs Outfield players

* Model Training - Match and Play Insights
  * Graph Conv Net?

### Approach Two - Spatio-Temporal Video Analysis

Just based on video and pixels, no skeleton based analysis.

## Research and Development

* Stack
  * Docker: Use docker to seemlessly transition from Dev environment to deployment environment, enabling consistency between mac and windows dev environemnts.
  * PyTorch: Seems like PyTorch is the favored deep learning library for python.
  * TensorFlow & Keras: Powerful for tuning and training on pre-existing models, strong for production.
  * CUDA: **Limited to INVIDIA ONLY. Leveraging GPU based compute power will be key in keeping runtimes short, although we may not acheive live processing speeds, we may want to take this route for something like a live broadcasting platform.

* Licensing
  * Trying to not use any paid-models, assess different models to see if this is advantageous.
  * Apache 2.0, MIT, or BSD licenses.

* Research
  * [Deep Learning for Videos: A 2018 Guide to Action Recognition](https://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)

* Data
  * [Kaggle - European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer/data?select=database.sqlite)
  * [Kaggle - FOOTBALL-SOCCER-VIDEOS-DATASET](https://www.kaggle.com/datasets/shreyamainkar/football-soccer-videos-dataset/data?status=pending)
  * [Kaggle - English Premier League](https://www.kaggle.com/datasets/saife245/english-premier-league)

* Tooling
  * PyTorch Geometric Tempory + PyTorch Lightning
  * [PyTorch3D by facebook](https://github.com/facebookresearch/pytorch3d/tree/main)
  * [TIMM](https://huggingface.co/docs/timm/index) - A pip package containing SOTA CV models
  * Scikit-learn - "classical machine learning"
  * [pytorch-recipies](https://github.com/facebookresearch/recipes)

* Computer Vision
  * Object Detection & Segmentation:
    * Mask R-CNN (slow but accurate)
    * FAIR's Detectron2 + Segment-Anything
    * OpenMMLab's MMDetection, MMSegmentation, MMPose
    * Google's MediaPipe
  * COCO:
    * [COCO 2020 Keypoint Detection Task](https://cocodataset.org/#keypoints-2020)
    * [COCO 2020 DensePose Task](https://cocodataset.org/#densepose-2020)
    * [COCO Dataset: All You Need to Know to Get Started](https://www.v7labs.com/blog/coco-dataset-guide)

* Models and Datasets
  * [Model Zoo](https://modelzoo.co/)
  * [HuggingFace](https://huggingface.co/)
  * [Kagle](https://www.kaggle.com/datasets)
  * [Sports M1](https://github.com/gtoderici/sports-1m-dataset/)
  * [Center for Research in Computer Vision](https://www.crcv.ucf.edu/data/UCF101.php)

* Papers
  * [Deep Learning for Videos: A 2018 Guide to Action Recognition](https://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)

* Learning resources:
  * [Udacity - DL w/ PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188)
  * [DAIR.AI](https://github.com/dair-ai)
    * [YouTube Universitry](https://github.com/dair-ai/ML-YouTube-Courses)
  * [PyTorch Notebooks](https://github.com/dair-ai/pytorch_notebooks)

* Data Annotation
  * [CVAT](https://www.cvat.ai/)
  * [CVAT via Docker](https://docs.cvat.ai/docs/administration/basics/installation/)

* Object Detection/Segmentation
  * [YOLO](https://github.com/hank-ai/darknet)

* Pose Estimation
  * [Facebook - SlowFast](https://github.com/facebookresearch/SlowFast)
  * [SuperGradients, YOLO-NAS](https://github.com/Deci-AI/super-gradients)
  * [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
  * [MMPose](https://github.com/open-mmlab/mmpose)
  * [RTMW3D: real-time model for 3D wholebody pose estimation](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d)
  * [RTMO: multi-person pose estimation](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmo)
  * [OpenPose](https://viso.ai/deep-learning/openpose/)
