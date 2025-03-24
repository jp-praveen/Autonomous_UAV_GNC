#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
import numpy as np
import joblib
from std_msgs.msg import Float32MultiArray

class NeuralNetworkNode:
    def __init__(self):
        rospy.init_node('neural_network_node', anonymous=True)

        # Load both models
        rospy.loginfo("Loading models...")

        self.model = self.rebuild_model('best_hyperparameters_min_2obs_1ofit.pkl', 11, 10)
        self.model.load_state_dict(torch.load('trained_model_min1_2obs_ofit.pth', map_location=torch.device('cpu')))
        self.model.eval()

        self.model_parTraj = self.rebuild_model('best_hyperparameters_parTrajFull_2obs_1ofit.pkl', 9, 14)
        self.model_parTraj.load_state_dict(torch.load('trained_model_parTrajFull_2obs_ofit.pth', map_location=torch.device('cpu')))
        self.model_parTraj.eval()

        rospy.loginfo("Models loaded successfully!")

        # ROS Publishers and Subscribers
        self.input_sub = rospy.Subscriber('/nn_input', Float32MultiArray, self.input_callback)
        self.output_pub = rospy.Publisher('/nn_output', Float32MultiArray, queue_size=10)

    def rebuild_model(self, hyperparams_file, input_size, output_size):
        """Rebuilds a model using hyperparameters from a saved file."""
        best_hyperparameters = joblib.load(hyperparams_file)
        n_layers = best_hyperparameters['n_layers']
        hidden_size = best_hyperparameters['hidden_size']
        activation_name = best_hyperparameters['activation']
        dropout_rate = best_hyperparameters['dropout_rate']

        # Define the activation function
        activation = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU()
        }.get(activation_name, nn.ReLU())

        # Build model layers
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))

        return nn.Sequential(*layers)

    def correct_waypoints(self, pos1, pos2, input_data):
        """Applies correction to the waypoints based on obstacle avoidance constraints."""
        x_obs1, y_obs1, r_obs1 = input_data[4], input_data[5], input_data[6]
        x_obs2, y_obs2, r_obs2 = input_data[7], input_data[8], input_data[9]

        # Correct pos1 if within obstacle 1 radius
        dist_x1 = abs(pos1[0] - x_obs1)
        if dist_x1 < r_obs1:
            pos1[0] += np.sign(pos1[0] - x_obs1) * (r_obs1 - dist_x1 + 0.3)

        # Correct pos2 if within obstacle 2 radius
        dist_x2 = abs(pos2[0] - x_obs2)
        if dist_x2 < r_obs2:
            pos2[0] += np.sign(pos2[0] - x_obs2) * (r_obs2 - dist_x2+0.3)

        return pos1, pos2

    def input_callback(self, msg):
        """Processes input, runs inference through both networks, applies corrections, and publishes outputs."""
        rospy.loginfo("Received input: {}".format(msg.data))

        # Convert ROS message to PyTorch tensor
        input_data = torch.tensor(msg.data).float().unsqueeze(0)

        # Pass through the first NN (model)
        with torch.no_grad():
            output_data = self.model(input_data).numpy()[0]

        # Extract waypoints and times from the first model
        pos0 = [msg.data[0], msg.data[1], 1]  # Initial position
        pos1 = [output_data[0], output_data[1], 1]  # First waypoint
        vel1 = [output_data[2], output_data[3]]  # Velocity at waypoint 1
        t1 = output_data[4]

        pos2 = [output_data[5], output_data[6], 1]  # Second waypoint
        vel2 = [output_data[7], output_data[8]]  # Velocity at waypoint 2
        t2 = output_data[9]

        pos3 = [msg.data[2], msg.data[3], 1]  # Final position
        t3 = msg.data[10]  # Final time

        # Apply correction step
        pos1, pos2 = self.correct_waypoints(pos1, pos2, msg.data)

        # Prepare inputs for second NN (model_parTraj)
        inputParTraj1 = torch.tensor([pos0[0], pos0[1], 0.0, 0.0, pos1[0], pos1[1], vel1[0], vel1[1], t1]).float()
        inputParTraj2 = torch.tensor([pos1[0], pos1[1], vel1[0], vel1[1], pos2[0], pos2[1], vel2[0], vel2[1], t2 - t1]).float()
        inputParTraj3 = torch.tensor([pos2[0], pos2[1], vel2[0], vel2[1], pos3[0], pos3[1], 0.0, 0.0, t3 - t2]).float()

        # Pass through the second NN (model_parTraj)
        with torch.no_grad():
            outputParTraj1 = self.model_parTraj(inputParTraj1).numpy()
            outputParTraj2 = self.model_parTraj(inputParTraj2).numpy()
            outputParTraj3 = self.model_parTraj(inputParTraj3).numpy()

        rospy.loginfo("Publishing corrected ParTraj outputs...")

        # Publish all three trajectory segments
        output_msg = Float32MultiArray()
        output_msg.data = np.concatenate([outputParTraj1, outputParTraj2, outputParTraj3]).tolist()
        self.output_pub.publish(output_msg)

        rospy.loginfo("Published corrected trajectory parameters.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        nn_node = NeuralNetworkNode()
        nn_node.run()
    except rospy.ROSInterruptException:
        pass
