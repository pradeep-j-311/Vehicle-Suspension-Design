# Vehicle Suspenion Design
A python library to help calculate the design calculations for a large omni-directional vehicle and its suspension characteristics. In order to design the vehicle, the dynamics  need to be computed from which the component specifications can be derived. The following state the fundamental properties of the vehicle: 

- 4-wheel drive 
- Electric vehicle (2 motors at each of the wheel, total of 8 motors)
- At each wheel, there are 2 degrees of freedom namely the drive motion and the steer motion 

At the start of the design phase, the electric motor characteristics need to be defined. For this several dynamic situations need to be considered such as slope, braking, acceleration, rolling resistance, coefficient of friction of the road and so on. This library helps to specify the characteristics of the motor that is needed to support the proposed vehicle parameters. Functions that are available in the library are as follows: 

- Calculate torque required at wheels for different dynamic scenarios 
- Load transfer and wheel load computations
- Suspension specifications based on encountered scenarios 
- Torsion bar length and diameter calculator based on suspension characteristics 
- Step response of vehicle with different models, such as: 
    - Quarter-car model 
    - Half-car pitch model 

The following sections detail out the calculations used in the python library. 

## Torque computations

$$ -(F_{r} + F_{f}) - \mu_{R}R_{f} - \mu_{R}R_{r} + mg\sin (\theta) = F $$ 

