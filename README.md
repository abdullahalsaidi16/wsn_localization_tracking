# Localization and Tracking in Wireless Sensors Network

The localization of moving targets in wireless sensor networks has been a very important problem in the past decade. Massive research has been dedicated to find solutions for this problem. Though there exist various methods to solve the problem, the Received Signal Level (RSL) method has proved to be one of the best. Technically this methods uses the signal strength levels from different anchor nodes as indicators to the corresponding position. However, building a mathematical model to link the position with the corresponding RSSLs is a very complicated impirical approach. For this reason we use machine learning algorithms to build the required mathematical model. Considering the tracking problem of moving targets to a be a successive application of localization algorithms, we make use of Kalman Filter and information available from the accelorometer of the moving target in order to obtain better accuracy. 
This project and the associated codes are part of my graduation project named "Localization and Tracking of a moving target in a wireless sensor network" in the department of Computer and Automation Engineering, Faculty of Mechanical and Electrical Engineering, Damasus University. 

## Getting Started

Simulation environment:
- Area: 200X200 m
- No. of Anchor nodes: 16
- No. of training examples: 100
The motion models we invetigated:
- Model of order 1
- Model of order 2
- Model of order 3
The ML Algorithms we used:
- SVM

### Prerequisites

This a python scripts need the following libraries to run them:
- scipy
- numpy
- scikit-learn
- matplotlib

## Functionality
you will see multiple scripts and files and I will interduce you to them:
First:
- [Lib](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/Lib) This is Folder containes my programming implementation for Kalman Filter in the Three orders
- [model](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/model) This contains script Model.py I used for Tunning the paramater of My SVM regressor and saving it also include a Lib folder
- [Random Fourier Features](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/Random%20Fourier%20Features) This folder also for testing techinique I used to know about the best parameters for RFF
- [Results](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/Results) This contains some importatnt images I saved for the project
- [trajectory](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/trajectory) This script used for building the trajectories for my study
- [WSN](https://github.com/abdullahalsaidi16/wsn_localization_tracking/tree/master/WSN) This scrpit very important for building the WSN rssi , Training point and Locating the sensors and I used as a function to be used in other places
- [main.py](https://github.com/abdullahalsaidi16/wsn_localization_tracking/blob/master/%20main.py) contains my study in the network and testing the perfomance and accurency for algorithms I used and shows how could be very useful.
All my results are plotted from this script.you 
- [README.md](https://github.com/abdullahalsaidi16/wsn_localization_tracking/blob/master/README.md) This the file you are reading :3 


## Contributing





## Authors

* **Abdullah Alsaidi** - *Initial work* - [Abdullah Alsaidi](https://github.com/abdullahalsaidi16)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License



## Acknowledgments

