- # README

  When people first started with Object Detection Task, the idea was to complete the task in two stages:

  - propose interesting regions to perform detection
  - and actually detect within the proposed regions

  While the accuracy of these algorithms are generally pretty awesome, the drawback of these approaches being that the 2-step-strategy is computationally very expensive for real-time applications.

  Single-Shot Detection(SSD) along with YOLO(v3) are two algorithms that excel with decent accuracy and extremely fast speed. 

  Acknowledgement: This is another project heavily guided by [sgrvinod’s](https://github.com/sgrvinod) tutorials. A big shout out for sgrvinod and his fantastic guides for learning various deep learning techniques. This project also depends on a number of really insightful academic papers such as: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf), [Scalable Object Detection using Deep Neural Networks](https://arxiv.org/pdf/1512.02325.pdf) and [Non-Maximum suppression](https://arxiv.org/pdf/1705.02950.pdf) just to name a few.

  ## An overview of SSD

  Single-Shot Detection has three main components:

  - **Base convolutions** derived from popular image classification architectures that provides the *lower-level features* that we need. Usually a **VGG-16** net is used here(not ResNet50 for faster computation)
  - **Auxiliary convolutions** added on top of the base network that will provide higher-level feature maps
  - **Prediction convolutions** that will locate and identify objects in these features maps

  sgrvinod provides excellent explanations of the basic features of SSD in [his repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). Do go and check it out!  Here are the main gits:

  - The task of detection is quite similar to classification tasks in that we need to figure out what object is in each sub-region. Object detection task, of course, focus more on the local detection.
  - Similar to how Jeremy Howard introduced in his FastAI series, we leverage **Transfer Learning** to obtain good results in shorter amount  of time.
  - Below is out base-architecture, VGG-16:

  ![vgg-16](images/vgg16.PNG)

  - however