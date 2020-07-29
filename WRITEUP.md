# Project Write-Up

I downloaded the model using this command:

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Next I extracted the tar.gz file using:
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Then I navigated to the extracted model directory using:
cd faster_rcnn_inception_v2_coco_2018_01_28

Converting the TensorFlow model to Intermediate Representation (IR) or OpenVINO IR format. The command used is given below:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

Next I used the "source env" button to initialize the OpenVino environment and install necessary libraries

Then I navigated to the workspace directory using:
cd /home/workspace
Then I started the Mosca server using:
cd webservice/server/node-server
node ./server.js

Next in a new terminal I started the GUI using:

cd webservice/ui
npm run dev

In another terminal, I ran the command:
sudo ffserver -f ./ffmpeg/server.conf

Next in a new terminal, to source OpenVino environment:
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

To run the app, I ran the code:
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

Then I clicked the Open App button

## Explaining Custom Layers

The process behind converting custom layers involves using the Model Optimizer to extract information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.

Some of the potential reasons for handling custom layers are: 1. Due to various frameworks which are used for training the deep learning models such as Keras, Tensorflow, ONNX, Caffe etc. and their different methods for processing data, some functions might not be available. The Custom Layer provides necessary support for operations not supported.


## Comparing Model Performance 

The size of the model pre-conversion was 142.21mb and it reduced afteer conversion. 
Due to the optimized Intermediate Representation of the model inference time is reduced by about 8%.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

1. Maintaining social distancing in public areas to avoid spread of Covid-19
2. Used during elections to maintain order when casting votes in the ballot box and also determining number of votes cast.
3. Finding the average duration of time spent by coustmers in a store.

Each of these use cases would be useful because it would help increase efficiency in surveilance and get relevant data in an effective manner.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

Poor lighting would reduce the accuracy of the model as the model performs better in environments with adequate lighting.

The accuracy of a deployed model determines how accurate predictions would be. In certain scenarios where the end-user has a device with minimal resources, we can make a trade-off between model accuracy and latency/computation speed.

Image size should be made to align with the required input size for the deployed model.
