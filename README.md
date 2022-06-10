# COMP4098_Supplementary_Material_Submission
COMP4098 Enhanced MSc Dissertation in Computer Science (Artificial Intelligence) Supplementary Material Submission 2021/2022

# spot_ros_autonomy
## UON - MSc COMP4098 Enhanced MSc Dissertation in Computer Science (Artificial Intelligence) 2021/2022

# Credits
Spot Base Mechanics: Chvmp/Champ (https://github.com/chvmp/champ) by Author: Juan Miguel Jimeno (https://github.com/grassjelly)

URDF: Clearpath Robotics (https://github.com/clearpathrobotics/spot_ros) by Clearpath Robotics (https://clearpathrobotics.com/)

## Material Submission 10/06/2022
Work created by the author:

- spot_ros_autonomy/
  - images/
    - A_001.png
    - B_001.png  
    - C_001.png
    - D_001.png
  - launch/
    - darknet_tobii.launch
    - OBJ_DETECT.launch
    - SAR_AI.launch
    - SAR_NORMAL.launch
    - TOBII.launch
    - TRAIN.launch
    - YOLO_FRONT.launch
  - maps/
    - map.pgm
    - map.yaml
  - src/
    - FLC.R
    - fuzzy_logic_controller_node.py
    - object_detection_node.py
    - qr_code_overlay.py
    - reset.py
    - spot_train_controller.py
    - tobii_controller_node.py
  - worlds/
    - SAR_AI.world
    - SAR.world
    - training_world.world
- spot_description/
  - urdf/
    - spot.urdf
- darknet_ros/
  - darknet_ros/
    - config/
      - backup/*
      - ros_front.yaml
      - yolov4_tiny_front.yaml
      - yolovr_tiny_tobii.yaml
    - launch/
      - backup/*
      - darknet_ros_front.launch
      - darknet_ros_tobii.launch


If building this project please follow instructions for installation of Champ by (https://github.com/chvmp/champ)

Clone this project repo to ~/workspace/src/

Run catkin_make & source devel/setup.bash

# System Specifications
Ubuntu - 18.04.06

Kernel - Tested on 5.11.0

ROS - Melodic

Gazebo - ver. 9

# Tested on System Hardware:

## System 1

  CPU - Intel i7 11800H @ 4.6GHz
  
  GPU - Nvidia RTX 3050ti (Mobile)
  
  RAM - 32GB
  
  Real-Time-Factor Avg: 1


