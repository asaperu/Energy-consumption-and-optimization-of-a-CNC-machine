# Energy-consumption-and-optimization-of-a-CNC-machine

This repository contains the Python code developed for a Master’s thesis focused on
the modelling, prediction, and optimization of energy consumption in CNC machining
operations.

The framework integrates G-code parsing, machine kinematics, and data-driven models
to estimate power consumption, processing time, cost, and environmental impact.
A web-based interface built with FastAPI allows users to upload CNC programs and
interactively evaluate machining performance under different operating conditions.


The repository Cloudflare API contains all the necessary to run the tunnel and access the website where to simulate the G-Code

The repository Extract_Data contains code to construct database between G-Code and Telemetry

The Input repository constitutes fictious yet functional data to use all of the programs

The Optimization repository contains a program that find the best k

The Training ANN has all the training performed to obtain the neural network architechture
