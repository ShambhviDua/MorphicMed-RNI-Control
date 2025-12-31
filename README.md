# MorphicMed-RNI-Control ->Code for Morphic-Med RNI-DNN temperature reconstruction and PID control logic.
#Curr control for the PCL-GNP kirigami sheet.

#Status:
#RNI Model: Converging (~0.92 R^2) but getting some noise spikes around Epoch 40.
#Hardware: MUX reading is currently blocking, need to move to threaded reading or C++ backend if latency gets worse.
#Sensor 142 on the board is dead, patched in pi_controller_main.py.

#Structure
#train_rni_v3.py: Main training loop. Expects the COMSOL .npy file in data/.
#pi_controller_main.py: The PID + Inference loop. DO NOT RUN WITHOUT SAFETY GOGGLES :D
#checkpoints/: Model weights saved every 10 epochs.

#To-Do
#Fix the integral windup in the PID (overshoots by 2 degrees sometimes).
#Re-export COMSOL data with finer mesh at the hinge vertices.

#Clean up the gradient clipping code.
