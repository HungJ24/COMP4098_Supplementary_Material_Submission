#----------------------------------------------------------------------------
#
# Created by: Victor Zhi Heung Ngo
# Created Date: March 2022
# Email: psxvn1@nottingham.ac.uk
#
#----------------------------------------------------------------------------
# Details about the module and for what purpose it was built for
#----------------------------------------------------------------------------
#
# Tested on Ubuntu 18.04, kernel 5.11, ROS Melodic
#
# Where specificed using the '#---' annotation, the author of the original 
# work is referenced.
#
# ---------------------------------------------------------------------------


# Load package FuzzyR
library(FuzzyR)

#################################################
### OBSTACLE AVOIDANCE FUZZY INFERENCE SYSTEM ###
#################################################

# Create a new fis
fis_obs <- newfis("obstacle_avoidance_FIS");

fis_obs <- addvar(fis_obs, "input", "obstacle_front", c(0, 15)); # distance in meters from robot
fis_obs <- addvar(fis_obs, "input", "obstacle_rear", c(0, 15));
fis_obs <- addvar(fis_obs, "input", "obstacle_left", c(0, 15));
fis_obs <- addvar(fis_obs, "input", "obstacle_right", c(0, 15));
fis_obs <- addvar(fis_obs, "output", "direction of movement", c(0, 10)); #crisp output value of robot"s state of direction/control

fis_obs <- addmf(fis_obs, "input", 1, "close", "trapmf", c(0, 0, 0.4, 0.6));
fis_obs <- addmf(fis_obs, "input", 1, "near", "trimf", c(0.4, 0.7, 1));
fis_obs <- addmf(fis_obs, "input", 1, "far", "trapmf", c(0.6, 0.75, 15, 15));

fis_obs <- addmf(fis_obs, "input", 2, "close", "trapmf", c(0, 0, 0.5, 0.6));
fis_obs <- addmf(fis_obs, "input", 2, "near", "trimf", c(0.4, 0.7, 1));
fis_obs <- addmf(fis_obs, "input", 2, "far", "trapmf", c(0.8, 3, 15, 15));

fis_obs <- addmf(fis_obs, "input", 3, "close", "trapmf", c(0, 0, 0.3, 0.5));
fis_obs <- addmf(fis_obs, "input", 3, "near", "trimf", c(0.2, 0.5, 1));
fis_obs <- addmf(fis_obs, "input", 3, "far", "trapmf", c(0.8, 3, 15, 15));

fis_obs <- addmf(fis_obs, "input", 4, "close", "trapmf", c(0, 0, 0.3, 0.5));
fis_obs <- addmf(fis_obs, "input", 4, "near", "trimf", c(0.2, 0.5, 1));
fis_obs <- addmf(fis_obs, "input", 4, "far", "trapmf", c(0.8, 3, 15, 15));

fis_obs <- addmf(fis_obs, "output", 1, "forward", "gaussmf", c(0.2, 0));
fis_obs <- addmf(fis_obs, "output", 1, "reverse", "gaussmf", c(0.2, 1));
fis_obs <- addmf(fis_obs, "output", 1, "left", "gaussmf", c(0.2, 2));
fis_obs <- addmf(fis_obs, "output", 1, "right", "gaussmf", c(0.2, 3));
fis_obs <- addmf(fis_obs, "output", 1, "left_rotate", "gaussmf", c(0.2, 4));
fis_obs <- addmf(fis_obs, "output", 1, "right_rotate", "gaussmf", c(0.2, 5));
fis_obs <- addmf(fis_obs, "output", 1, "forward_left_diagonal", "gaussmf", c(0.2, 6));
fis_obs <- addmf(fis_obs, "output", 1, "forward_right_diagonal", "gaussmf", c(0.2, 7));
fis_obs <- addmf(fis_obs, "output", 1, "reverse_left_diagonal", "gaussmf", c(0.2, 8));
fis_obs <- addmf(fis_obs, "output", 1, "reverse_right_diagonal", "gaussmf", c(0.2, 9));
fis_obs <- addmf(fis_obs, "output", 1, "stop", "gaussmf", c(0.2, 10));

# Defines the rule list for the fis_obs model.
# Example: "1, 3, 3, 3, 2, 1, 1"
# Col 1 = 1, front obstacle distance input
# Col 2 = 3, rear obstacle distance input
# Col 3 = 3, left obstacle distance input
# Col 4 = 3, right obstacle distance input
# Col 5 = 2, direction of movement output
# Col 5 = 1, The weight applied to the rule
# Col 6 = 1, Fuzzy Operator AND

rulelist <- rbind(c(1, 3, 3, 3, 2, 1, 1), c(3, 1, 3, 3, 1, 1, 1), 
                  c(3, 3, 1, 3, 4, 1, 1), c(3, 3, 3, 1, 3, 1, 1), 
                  c(1, 1, 3, 3, 6, 1, 1), c(1, 3, 1, 3, 10, 1, 1), 
                  c(1, 3, 3, 1, 9, 1, 1), c(3, 3, 1, 1, 1, 1, 1), 
                  c(1, 1, 3, 1, 5, 1, 1), c(1, 1, 1, 3, 6, 1, 1), 
                  c(1, 3, 1, 1, 2, 1, 1), c(3, 1, 1, 1, 1, 1, 1), 
                  c(3, 1, 1, 3, 8, 1, 1), c(3, 1, 3, 1, 7, 1, 1), 
                  c(1, 2, 2, 2, 2, 1, 1), c(2, 1, 2, 2, 1, 1, 1), 
                  c(2, 2, 1, 2, 4, 1, 1), c(2, 2, 2, 1, 3, 1, 1), 
                  c(1, 1, 2, 2, 6, 1, 1), c(1, 2, 1, 2, 10, 1, 1), 
                  c(1, 2, 2, 1, 9, 1, 1), c(2, 2, 1, 1, 1, 1, 1), 
                  c(1, 1, 2, 1, 5, 1, 1), c(1, 1, 1, 2, 6, 1, 1), 
                  c(1, 2, 1, 1, 2, 1, 1), c(2, 1, 1, 1, 1, 1, 1), 
                  c(2, 1, 1, 2, 8, 1, 1), c(2, 1, 2, 1, 7, 1, 1), 
                  c(1, 2, 3, 3, 2, 1, 1), c(1, 2, 2, 3, 2, 1, 1), 
                  c(1, 3, 2, 3, 2, 1, 1), c(1, 3, 2, 2, 2, 1, 1), 
                  c(1, 3, 3, 2, 2, 1, 1), c(2, 1, 3, 3, 1, 1, 1), 
                  c(2, 1, 2, 3, 1, 1, 1), c(3, 1, 2, 3, 1, 1, 1), 
                  c(3, 1, 2, 2, 1, 1, 1), c(3, 1, 3, 2, 1, 1, 1), 
                  c(2, 3, 1, 3, 4, 1, 1), c(2, 2, 1, 3, 4, 1, 1), 
                  c(3, 2, 1, 3, 4, 1, 1), c(3, 2, 1, 2, 4, 1, 1), 
                  c(3, 3, 1, 2, 4, 1, 1), c(2, 3, 3, 1, 3, 1, 1), 
                  c(2, 2, 3, 1, 3, 1, 1), c(3, 2, 3, 1, 3, 1, 1), 
                  c(3, 2, 2, 1, 3, 1, 1), c(3, 3, 2, 1, 3, 1, 1), 
                  c(1, 1, 2, 3, 6, 1, 1), c(1, 1, 3, 2, 6, 1, 1), 
                  c(1, 2, 1, 3, 10, 1, 1), c(1, 3, 1, 2, 10, 1, 1), 
                  c(1, 2, 3, 1, 9, 1, 1), c(1, 3, 2, 1, 9, 1, 1), 
                  c(2, 3, 1, 1, 1, 1, 1), c(3, 2, 1, 1, 1, 1, 1), 
                  c(2, 1, 1, 3, 8, 1, 1), c(3, 1, 1, 2, 8, 1, 1), 
                  c(2, 1, 3, 1, 7, 1, 1), c(3, 1, 2, 1, 7, 1, 1), 
                  c(1, 1, 1, 1, 11, 1, 1));


#Add the rule list to the model.
fis_obs <- addrule(fis_obs, rulelist);

################################################################
### ATTENTION SCORE & INTENTION ERROR FUZZY INFERENCE SYSTEM ###
################################################################

# Create a new fis object
fis_AI <- newfis("Attention and Intention FIS")

# Attention score (percentage of time gaze is in the screen boundaries)
fis_AI <- addvar(fis_AI, "input", "attention", c(0, 100));
# S.d. Error from the optimal path taken from ROS move_base
fis_AI <- addvar(fis_AI, "input", "intention_goal_error", c(0, 100));
# Level of Autonomy
fis_AI <- addvar(fis_AI, "output", "LOA", c(0, 1));

fis_AI <- addmf(fis_AI, "input", 1, "very_low", "gaussmf", c(15, 0));
fis_AI <- addmf(fis_AI, "input", 1, "low", "gaussmf", c(20, 30));
fis_AI <- addmf(fis_AI, "input", 1, "normal", "gaussmf", c(15, 60));
fis_AI <- addmf(fis_AI, "input", 1, "high", "gaussmf", c(10, 100));

fis_AI <- addmf(fis_AI, "input", 2, "very_low", "gaussmf", c(10, 0));
fis_AI <- addmf(fis_AI, "input", 2, "low", "gaussmf", c(10, 25));
fis_AI <- addmf(fis_AI, "input", 2, "medium", "gaussmf", c(15, 60));
fis_AI <- addmf(fis_AI, "input", 2, "high", "gaussmf", c(15, 100));

fis_AI <- addmf(fis_AI, "output", 1, "full_autonomy", "gaussmf", c(0.1, 0));
fis_AI <- addmf(fis_AI, "output", 1, "semi_autonomy", "gaussmf", c(0.15, 0.5));
fis_AI <- addmf(fis_AI, "output", 1, "full_teleop", "gaussmf", c(0.1, 1));

#fis_AI <- addmf(fis_AI, "output", 1, "full_autonomy", "singletonmf", (0));
#fis_AI <- addmf(fis_AI, "output", 1, "semi_autonomy", "singletonmf", (0.5));
#fis_AI <- addmf(fis_AI, "output", 1, "full_teleop", "singletonmf", (1));

rulelist_AI <- rbind(c(1, 1, 1, 1, 1), c(1, 2, 1, 1, 1),
                     c(1, 3, 1, 1, 1), c(1, 4, 1, 1, 1),
                     c(2, 1, 2, 1, 1), c(2, 2, 2, 1, 1),
                     c(2, 3, 2, 1, 1), c(2, 4, 2, 1, 1),
                     c(3, 1, 3, 1, 1), c(3, 2, 3, 1, 1),
                     c(3, 3, 3, 1, 1), c(3, 4, 3, 1, 1),
                     c(4, 1, 3, 1, 1), c(4, 2, 3, 1, 1),
                     c(4, 3, 3, 1, 1), c(4, 4, 3, 1, 1));
                    
fis_AI <- addrule(fis_AI, rulelist_AI)

#################
### FUNCTIONS ###
#################

return_obs_eval_fis <- function(front_dist, rear_dist, left_dist, right_dist) {
  obs_evaluation <- evalfis(c(front_dist, rear_dist, left_dist, right_dist), fis_obs)
  return(obs_evaluation)
}

return_loa_eval_fis <- function(att_score, intent_err) {
  loa_evaluation <- evalfis(c(att_score, intent_err), fis_AI)
  return(loa_evaluation)
}

gen_plots <- function() {
  # Resize the graph window output to an appropriate size.
  par(mfrow = c(1, 1));

  # Plots the graphs for membership functions of Attention, Intention Goal Error and LOA.
  plotmf(fis_AI, "input", 1, main = "Attention Membership Fucntions");
  plotmf(fis_AI, "input", 2, main = "Intention Goal Error Membership Functions");
  plotmf(fis_AI, "output", 1, main = "State of LOA Membership Functions");

  gensurf(fis_AI)
}

gen_plots_obs <- function() {
  # Resize the graph window output to an appropriate size.
  par(mfrow = c(1, 1));

  # Plots the graphs for membership functions of each sensor and movement.
  plotmf(fis_obs, "input", 1, main = "3 Term obstacle_front function plot");
  plotmf(fis_obs, "input", 2, main = "3 Term obstacle_rear function plot");
  plotmf(fis_obs, "input", 3, main = "3 Term obstacle_left function plot");
  plotmf(fis_obs, "input", 4, main = "3 Term obstacle_right function plot");
  plotmf(fis_obs, "output", 1, main = "12 Term Direction of Movement function plot");
}