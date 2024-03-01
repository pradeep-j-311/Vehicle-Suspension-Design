
# %%

import numpy as np 
import math as m
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

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
    
    def suspension_specs(self, accel_g, Nmm = False): 
        # the spring rate for the suspension will be determined by the maxsiumum travel allowed for the suspension under maxsiumum load 
        maxs_load = maxs(self.load_transfer(accel_g))[0]
        min_load = min(self.load_transfer(accel_g, loaded=False))[0]
        wheel_rate = (maxs_load - min_load)/self.travel_length

        if not Nmm: 
            return wheel_rate
        else: 
            return wheel_rate/1000
        
    def torsion_bar_solid(self, wheel_load, spring_rate=161*10**3, shear_modulus=8*10**10, allowable_stress=7*10**9, lever_arm=0.02, front=True): 
        # angle of twist = TL/JG
        # for a circular shaft we J = pi.d^4/32 (solid shaft)
       
        A = allowable_stress/(16*wheel_load)
        diameter = (lever_arm/(A*np.pi))**(1/3)
        B = 0.098*shear_modulus/spring_rate
        length = (B*diameter**4)/lever_arm**2

        return diameter, length, wheel_load
    
    def quarter_car_model_step(self, t, x, m1, m2, damping_rate, spring_rate, tire_spring_rate, road_input, step_time): 
        # quarter car mode assumes two masses (body and tire)
        # tire has a spring, body has spring and damper 
        x1_dot, x1, x2_dot, x2 = x

        # step action
        if t >= step_time: 
            y = road_input
        else: 
            y = 0

        vel = np.array([[x1_dot], [x2_dot]])
        pos = np.array([[x1], [x2]])
        M = np.array([[m1, 0], [0, m2]])
        C = np.array([[damping_rate, -damping_rate], [-damping_rate, damping_rate]])
        K = np.array([[spring_rate, -spring_rate], [-spring_rate, spring_rate+tire_spring_rate]])
        F = np.array([[0], [tire_spring_rate*y]])
        acc = np.matmul(np.linalg.inv(M), -np.matmul(C, vel) - np.matmul(K, pos) + F)
        x_dot = np.array([acc[0, 0], x1_dot, acc[1, 0], x2_dot])

        return x_dot
    
    def init_parameters_quarter_car(self): 
        m1 = self.auc_mass/4 - self.unsprung_mass
        m2 = self.unsprung_mass
        spring_rate=129*10**3
        tire_spring_rate=350*10**3
        damping_rate = 2*m.sqrt(spring_rate*(self.auc_mass/4 - self.unsprung_mass))
        road_input=1
        step_time=0
        t_span=(0, 5)
        x0 = np.array([0, 0, 0, 0])

        parameters = (m1, m2, damping_rate, spring_rate, tire_spring_rate, road_input, step_time)
        inCon = [t_span, x0]
        
        return parameters, inCon
    
    def plot_step(self, sol): 
        with plt.xkcd(): 
            plt.figure()
            plt.plot(sol.t, sol.y[1, :], 'b--', sol.t, sol.y[3, :], 'r--')
            plt.grid(True)
            plt.xlabel('time (s)')
            plt.ylabel('Displacement (m)')
            plt.title('Step reponse of a quarter car model')
            plt.show()
            
    def hollow_torsion_bar(self, wheel_load, OD, ID, allowable_shear_stress=7*10**9, lever_arm=0.05, shear_modulus=80*10**9, spring_rate=161*10**3): 
        J = np.pi*(OD**4 - ID**4)/2
        # maxs shear stress will occur on the OD 
        maxs_torque = allowable_shear_stress*J/lever_arm  
        applied_torsion = wheel_load*lever_arm 
        
        if applied_torsion > maxs_torque: 
            return "Applied torsion exceeds maxsimum recommended torsion on chosen bar."
        else: 
            L = (shear_modulus*np.pi*(OD**4 - ID**4))/(spring_rate*16*lever_arm**2)
            return L 

    def pitch_car_model_step(self, t, x, mBody, m1, m2, Iz, damping_rates, spring_rates, tire_spring_rates, road_input, step_time, speed): 
        
        xBody, xBody_dot, theta, thetadot, x1, x1_dot, x2, x2_dot = x
        c1, c2 = damping_rates
        k1, k2 = spring_rates
        kt1, kt2 = tire_spring_rates
        a1 = self.wheelbase - self.front_weight_bias*self.wheelbase
        a2 = self.wheelbase - a1
        
        # step input for the pitch model (varies by speed)
        if t >= step_time: 
            y1 = road_input
        else: 
            y1 = 0 
        if t >= step_time + self.wheelbase/speed: 
            y2 = road_input
        else: 
            y2 = 0 
            
        M = np.array([[mBody, 0, 0, 0], [0, Iz, 0, 0], [0, 0, m1, 0], [0, 0, 0, m2]])
        C = np.array([[c1+c2, (a2*c2) - (a1*c1), -c1, -c2], 
                      [(a2*c2)-(a1*c1), c1*(a1**2) + c2*(a2**2), a1*c1, -a2*c2], 
                      [-c1, a1*c1, c1, 0], 
                     [-c2, -a2*c2, 0, c2]])
        K = np.array([[k1 + k2, (a2*k2) - (a1*k1), -k1, -k2],
                      [(a2*k2) - (a1*k1), k1*(a1**2) + k2*(a2**2), a1*k1, -a2*k2],
                      [-k1, a1*k1, k1 + kt1, 0],
                    [-k2, -a2*k2, 0, k2+kt2]])
        F = np.array([[0], [0], [y1*kt1], [y2*kt2]])
        
        pos = np.array([[xBody], [theta], [x1], [x2]])
        vel = np.array([[xBody_dot], [thetadot], [x1_dot], [x2_dot]])
        
        xddot = np.matmul(np.linalg.inv(M), -np.matmul(C, vel) - np.matmul(K, pos) + F)
        
        xDot = np.array([xBody_dot, xddot[0, 0], thetadot, xddot[1, 0], x1_dot, xddot[2, 0], x2_dot, xddot[3, 0]])
        
        return xDot 
        
    def init_parameters_pitch(self, speed, loaded = True): 
        kt1 = kt2 = 350*10**3
        k1 = k2 = 129*10**3 
        m1 = m2 = 50 
        if loaded: mBody = (self.auc_mass + self.uld_mass - 4*self.unsprung_mass)/2 
        else: mBody = (self.auc_mass -4*self.unsprung_mass)/2 
        c1 = c2 = m.sqrt(k1*mBody/2)*2*0.2
        Iz = (1/12)*mBody*((2*self.cg)**2 + self.wheelbase**2)
        road_input = 0.5 
        step_time = 0
        
        # gather inputs 
        damping_rates = [c1, c2]
        spring_rates = [k1, k2]
        tire_spring_rates = [kt1, kt2]
        
        # gather initial conditions 
        x0 = np.array([0,0,0,0,0,0,0,0])
        t_span = (0, 5)
        
        params = (mBody, m1, m2, Iz, damping_rates, spring_rates, tire_spring_rates, road_input, step_time, speed)
        inCon = [t_span, x0]
        
        return params, inCon 
    
    def plot_pitch_step(self, sol): 
        with plt.xkcd(): 
            fig, axs = plt.subplots(2, 2, figsize=(22, 15))
            
            axs[0,0].plot(sol.t, sol.y[0, :], 'k--', sol.t, sol.y[4, :], 'r--', sol.t, sol.y[6, :])
            axs[0,0].set_xlabel('time (s)')
            axs[0,0].set_ylabel('Displacement (m)')
            axs[0,0].set_title('Wheels/Body vertical step reponse of a half car pitch model')
            
            axs[0,1].plot(sol.t, sol.y[2], 'g--')
            axs[0,1].set_xlabel('time (s)')
            axs[0,1].set_ylabel('Body angle, (rad)')
            axs[0,1].set_title('Body angle step reponse of a half car pitch model')
            
            axs[1,0].plot(sol.t, sol.y[1, :], 'k--', sol.t, sol.y[5, :], 'r--', sol.t, sol.y[7, :], 'b--')
            axs[1,0].set_xlabel('time (s)')
            axs[1,0].set_title('Wheel/Body velocities step response of a half car pitch model')
            axs[1,0].set_ylabel('Velocity (m/s)')
            
            axs[1,1].plot(sol.t, sol.y[3, :], 'g--')
            axs[1,1].set_xlabel('time (s)')
            axs[1,1].set_ylabel('Angular velocity (rad/s)')
            axs[1,1].set_title('Body angular velocity step response of a half car pitch model')
            
            plt.show()

# %%
