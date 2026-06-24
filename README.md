# 2D-Particle-Filter
## Project Overview
The purpose of this project was to implement a 2D particle filter in MATLAB based on the 2002 paper "A Tutorial on Particle Filters for Online
Nonlinear/Non-Gaussian Bayesian Tracking" by Arulampalam et al.  The specific Particle Filter that was chosen to be implemented was the Sampling Importance Resampling Filter in order to prevent sample impoverishment and to efficiently evaluate the importance weights. 
## Mathematical Model
The non-linear discrete-time system is composed of two equations for the x and y coordinates:
```math
x_k = 0.5 x_{k-1} + \frac{25 x_{k-1}}{1 + x_{k-1}^2} + 8 \cos(1.2 k \Delta t) + 0.1 y_{k-1} + w_{x,k}
```
```math
y_k = 0.5 y_{k-1} + \frac{20 y_{k-1}}{1 + y_{k-1}^2} + 6 \sin(0.8 k \Delta t) + 0.1 x_{k-1} + w_{y,k}
```
The system was chosen to be measured by a GPS tracker, using direct Cartesian position tracking, for simplicity(though the measurement function could have been chosen to simulate a LiDAR sensor). Thus the  linear measurement function for the non-linear system is:
```math
z_k = [x_k,y_k]^T+ v_k
```
The $w_k$ and $v_k$ terms come from the process covariance matrix, Q, and measurement covariance matrix, R, respectively:
```math
{w}_k \sim \mathcal{N}(\mathbf{0}, Q),
Q = \begin{bmatrix} 1.0 & 0.1 \\ 0.1 & 0.8 \end{bmatrix}
```
```math
{v}_k \sim \mathcal{N}(\mathbf{0}, R),
R = \begin{bmatrix} 1.0 & 0 \\ 0 & 1.0 \end{bmatrix}
```
## How to Run
1. Clone this repository.
2. Open MATLAB and run `particle_filter_2d.m`.
3. The script will execute the 100-step, 50 particle simulation and output a performance summary figure displaying 2D tracking trajectories, effective particle sizes, tracking errors, and resampling frequencies.


