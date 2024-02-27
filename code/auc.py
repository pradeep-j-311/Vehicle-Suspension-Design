import numpy as np 
import math as m
import matplotlib.pyplot as plt 

class auc(): 
    def __init__(self): 
        # define AUC parameters 
        self.auc_mass = 2050
        self.uld_mass = 1588
        self.unsprung_mass = 50
        self.safety_factor = 1.3 
        self.rolling_resistance = 0.015 
        self.friction_coefficient = 0.4 
        self.cg = 1.3
        self.wheelbase = 3.0 
        self.offset = 0.14
        self.wheel_radius = 0.2
        self.g = 9.81
        self.roll_gear_ratio = 5.0
        self.steer_gear_ratio = 7.0
        self.i2rstuff = 50
        self.travel_length = 0.1
        self.front_weight_bias = 0.5

    def torque_required(self, slope, acceleration, loaded=False, mode="roll", message=False): 
        # change to radians
        slope = np.deg2rad(slope)
        #x = A^-1 * b 
        # A matrix
        A = np.array([[1, -self.rolling_resistance, -self.rolling_resistance], 
                      [self.cg, -self.rolling_resistance*(self.cg-self.wheel_radius) + self.wheelbase/2 - self.offset, -self.rolling_resistance*(self.cg-self.wheel_radius) - self.wheelbase/2 - self.offset], 
                      [0, 1, 1]])
        if loaded: 
            total_mass = self.safety_factor*self.auc_mass + self.uld_mass + self.i2rstuff
        else: 
            total_mass = self.safety_factor*self.auc_mass + self.i2rstuff

        F = total_mass*acceleration
        b = np.array([[F + total_mass*self.g*m.sin(slope)], [0], [total_mass*self.g*m.cos(slope)]])
        x = np.matmul(np.linalg.inv(A), b)

        # divide by two, since only considering one side 
        rear_wheel_load = x[2, 0]/2
        front_wheel_load = x[1, 0]/2
        acceleration_force = x[0, 0]/2
        wheel_loads = np.array([[front_wheel_load], [rear_wheel_load]])

        # compute wheel forces and torques 
        if mode=="roll": 
            front_motor_torque, front_wheel_force, rear_motor_torque, rear_wheel_force = self.wheel_torque(self.roll_gear_ratio, self.front_weight_bias, acceleration_force, self.wheel_radius)
        else: 
            front_motor_torque, front_wheel_force, rear_motor_torque, rear_wheel_force = self.wheel_torque(self.steer_gear_ratio, self.front_weight_bias, acceleration_force, self.offset)
        
        # check available force based on ground friction
        available_front_force = front_wheel_load*self.friction_coefficient
        available_rear_force = rear_wheel_load*self.friction_coefficient

        # check for wheel slippage 
        if (available_front_force < front_wheel_force) or (available_rear_force < rear_wheel_force): 
            slip = True 
        else: 
            slip = False

        # return result 
        if message: 
            if slip:
                return "Front motor torque: {}, Rear motor torque: {}. Wheel slip will occur.".format(front_motor_torque, rear_motor_torque)
            else: 
                return "Front motor torque: {}, Rear motor torque: {}.".format(front_motor_torque, rear_motor_torque)
        else: 
            return front_motor_torque, rear_motor_torque, wheel_loads, slip
        
    def wheel_torque(self, gear_ratio, bias, acceleration_force, radius): 

        front_wheel_force = bias*acceleration_force
        front_wheel_torque = front_wheel_force*radius
        front_motor_torque = front_wheel_torque/gear_ratio
        rear_wheel_force = (1-bias)*acceleration_force
        rear_wheel_torque = rear_wheel_force*radius
        rear_motor_torque = rear_wheel_torque/gear_ratio

        return front_motor_torque, front_wheel_force, rear_motor_torque, rear_wheel_force
    
    def load_transfer(self, accel_g, loaded=True):
        # function offers a quick way to compute wheel loads for acceleration over flat ground 
        if loaded: 
            accel_force = (self.uld_mass + self.auc_mass)*accel_g*9.81*self.safety_factor
        else: 
            accel_force = (self.auc_mass)*accel_g*9.81*self.safety_factor
        
        distance_to_front = self.front_weight_bias*self.wheelbase
        distance_to_rear = self.wheelbase - distance_to_front
        # build the simultaneous equation set up 
        A = np.array([[1, 1], [distance_to_front, -distance_to_rear]])
        b = np.array([[accel_force/accel_g], [-self.cg*accel_force]])
        wheel_loads = np.matmul(np.linalg.inv(A), b)
    
        return wheel_loads/2
    
    def suspension_specs(self, wheel_loads, Nmm = False): 
        # the spring rate for the suspension will be determined by the maxiumum travel allowed for the suspension under maxiumum load 
        max_load = max(wheel_loads)[0]
        wheel_rate = max_load/self.travel_length

        if not Nmm: 
            return wheel_rate
        else: 
            return wheel_rate/1000
        
    def torsion_bar(self, spring_rate=161*10**3, shear_modulus=8*10**10, allowable_stress=7*10**9, lever_arm=0.02, front=True): 
        # angle of twist = TL/JG
        # for a circular shaft we J = pi.d^4/32 (solid shaft)
        if front: 
            wheel_weight = 9.81*(self.auc_mass*self.safety_factor + self.uld_mass + self.i2rstuff)*self.front_weight_bias/2
        else: 
            wheel_weight = 9.81*(self.auc_mass*self.safety_factor + self.uld_mass + self.i2rstuff)*(1 - self.front_weight_bias)/2

        A = allowable_stress/(16*wheel_weight)
        diameter = (lever_arm/(A*np.pi))**(1/3)
        B = 0.098*shear_modulus/spring_rate
        length = (B*diameter**4)/lever_arm**2

        return diameter, length, wheel_weight
    
    def quarter_car_model(self, t, x, damping_rate, spring_rate, tire_spring_rate, road_input): 
        # quarter car mode assumes two masses (body and tire)
        # tire has a spring, body has spring and damper 
        m1 = self.auc_mass/4 - self.unsprung_mass
        m2 = self.unsprung_mass
        x1_dot =  x[0] 
        x1 = x[1]
        x2_dot= x[2]
        x2 = x[3]
        vel = np.array([[x1_dot], [x2_dot]])
        pos = np.array([[x1], [x2]])
        M = np.array([[m1, 0], [0, m2]])
        C = np.array([[damping_rate, -damping_rate], [-damping_rate, damping_rate]])
        K = np.array([[spring_rate, -spring_rate], [-spring_rate, spring_rate+tire_spring_rate]])
        F = np.array([[0], [tire_spring_rate*road_input]])
        acc = np.matmul(np.linalg.inv(M), -np.matmul(C, vel) - np.matmul(K, pos) + F)
        x_dot = np.array([acc[0, 0], acc[1, 0], acc[2, 0], acc[3, 0]])

        return x_dot