cmake_minimum_required(VERSION 2.8.3)
project(spot_ros_autonomy)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

catkin_install_python(PROGRAMS
   src/fuzzy_logic_controller_node.py
   src/object_detection_node.py
   src/tobii_controller_node.py
   src/spot_train_controller.py
   src/qr_code_overlay.py
   src/reset.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)


install(FILES
  launch/SAR_NORMAL.launch
  launch/SAR_AI.launch
  launch/YOLO_FRONT.launch
  launch/TOBII.launch
  launch/OBJ_DETECT.launch
  launch/TRAIN.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(DIRECTORY config launch maps src worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

