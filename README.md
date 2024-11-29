# FrameGate

https://blog.tensorflow.org/2022/04/video-classification-on-edge-devices.html

https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu

https://www.losant.com/blog/how-to-access-the-raspberry-pi-camera-in-docker

https://www.kaggle.com/models/google/movinet/tfLite/a0-stream-kinetics-600-classification-tflite-int8

https://paperswithcode.com/paper/movinets-mobile-video-networks-for-efficient 


https://www.jeffgeerling.com/blog/2023/testing-coral-tpu-accelerator-m2-or-pcie-docker

https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/raspberry_pi 

https://coral.ai/docs/edgetpu/models-intro/#model-requirements

https://www.tensorflow.org/tutorials/load_data/video

https://www.tensorflow.org/hub/tutorials/movinet

https://coral.ai/docs/edgetpu/models-intro/#compatibility-overview

https://coral.ai/docs/accelerator/get-started/#next-steps

https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/7

https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset

https://www.linuxfoundation.org/hubfs/Research%20Reports/Open_Source_and_Energy_Interoperability_2024_082124a.pdf?hsLang=en&__hstc=138353967.6a396ecf6cf2c4023ead0e8eab836040.1731076407581.1731076407581.1731076407581.1&__hssc=138353967.2.1731076407581&__hsfp=301312424

https://new.nsf.gov/news/new-material-computer-chips-could-reduce-energy#:~:text=It's%20estimated%20that%20the%20information,world's%20power%20generation%20by%202030.

https://www.sentinelone.com/cybersecurity-101/cybersecurity/cyber-security-statistics/

https://www.mybib.com/tools/ieee-citation-generator

https://arxiv.org/abs/2103.11511

https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet

https://github.com/tensorflow/models/tree/master/official/vision



https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb#scrollTo=joxrIB0I3cdi

Edge TPU Compiler version 16.0.384591198
Started a compilation timeout timer of 180 seconds.

Model compiled successfully in 141 ms.

Input model: 1.tflite
Input size: 5.11MiB
Output model: 1_edgetpu.tflite
Output size: 5.09MiB
On-chip memory used for caching model parameters: 18.00KiB
On-chip memory remaining for caching model parameters: 6.86MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
Number of Edge TPU subgraphs: 1
Total number of operations: 368
Operation log: 1_edgetpu.log

Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 15
Number of operations that will run on CPU: 353
See the operation log file for individual operation details.
Compilation child process completed within timeout period.
Compilation succeeded! 