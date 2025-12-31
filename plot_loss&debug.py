"""
RNI Real-Time Controller - "The Temperature Tamer"
Deployment version: takes trained model + PID and runs on actual hardware.
Last tested: 2025-12-18
Warning: This actually controls a LASER. Triple-check safety limits.
"""

import time
import numpy as np
import torch

# Import model architecture - must match training exactly
from train_rni_v3 import RNI_Net

# ============================================================================
# HARDWARE PIN MAPPING (BCM numbering - not board, not wiringPi)
# ============================================================================
LASER_PWM_PIN = 18    # Hardware PWM capable (only GPIO12, GPIO13, GPIO18, GPIO19)
MUX_SIG_PIN = 23      # Regular GPIO for MUX address lines
# ADC uses SPI0 (CE0) - handled by spidev if we ever get it working reliably

# ============================================================================
# PID TUNING (Final values after 3 days of oscillation hunting :z)
# ============================================================================
# Tuned on 2024-03-17 with actual thermal load (heatsink + fan setup)
Kp = 2.5    # Proportional: Aggressive to overcome thermal mass
Ki = 0.8    # Integral: Accumulates slowly - watch for windup!
Kd = 0.15   # Derivative: Damping term - too high causes jitter

TARGET_TEMP = 45.0      # °C - sweet spot for our material
SAFETY_LIMIT = 48.0     # °C - ABSOLUTE MAX before emergency shutdown
                        # Above 50°C and the epoxy starts to soften...


class HardwareInterface:
    """
    Talks to the actual hardware .
    
    Current hardware bugs:
    1. MUX channel 7 has crosstalk (addressed in read_all_sensors)
    2. ADC reference voltage drifts with temperature (not compensated)
    3. GPIO23 has weird pull-up behavior (addressed with explicit set)
    """
    
    def __init__(self):
        """
        Initialize hardware or simulation.
        
        IMPORTANT: Comment out the 'import' lines when running on laptop,
        but don't delete them or you'll forget to re-add them for deployment.
        """
        # ====================================================================
        # REAL IMPORTS (commented for development)
        # ====================================================================
        # import RPi.GPIO as GPIO
        # import spidev
        # self.GPIO = GPIO
        # self.spi = spidev.SpiDev()
        
        self.num_sensors = 308  # Fixed by PCB design (can't change without respin)
        
        print("[HARDWARE] Running in SIMULATION MODE")
        print("[HARDWARE] To run on real hardware:")
        print("[HARDWARE] 1. Uncomment GPIO/spidev imports")
        print("[HARDWARE] 2. Connect heatsink + fan")
        print("[HARDWARE]
    
    def read_all_sensors(self):
        """
        Reads all 308 voltage segments via 74HC4067 MUX + MCP3008 ADC.
        
        Timing: Currently ~40ms for full sweep (25kHz sampling).
        Problem: Need <20ms for stable 50Hz control loop.
        TODO: Switch to hardware-timed sampling or parallel ADCs.
        
        Returns:
            torch.Tensor of shape (1, 308) - batch dimension for model
        """
        # Simulated noisy ADC values (0.5-2.5V typical range)
        # Real voltages are ratiometric to 3.3V reference
        raw_voltages = np.random.uniform(0.5, 2.5, self.num_sensors)
        
        # ====================================================================
        # HARDWARE HACKS / WORKAROUNDS
        # ====================================================================
        # Sensor 142 is dead on Rev2 board (bad solder joint under QFN)
        # Linear interpolation from neighbors as temporary fix
        raw_voltages[142] = (raw_voltages[141] + raw_voltages[143]) / 2
        
        # MUX channel 7 has crosstalk from channel 6 (layout issue)
        # Apply correction factor measured during calibration
        raw_voltages[7::16] = raw_voltages[7::16] * 0.92
        
        # Add simulated ADC noise (LSB flipping)
        adc_noise = np.random.normal(0, 0.01, self.num_sensors)
        raw_voltages += adc_noise
        
        # Convert to tensor with batch dimension
        voltage_tensor = torch.FloatTensor(raw_voltages).unsqueeze(0)
        
        return voltage_tensor
    
    def set_laser(self, duty_cycle):
        """
        Sets laser PWM duty cycle (0-100%).
        
        Safety: Hardware limit at 90% in driver circuit, but we cap at 85%
        for additional safety margin. Laser diode lifetime drops exponentially
        above 80% duty.
        
        Args:
            duty_cycle: Float 0-100, will be clamped
        """
        # Hardware safety clamp
        duty_cycle = max(0.0, min(85.0, duty_cycle))  # 85% ABSOLUTE MAX
        
        # Real implementation:
        # self.GPIO.output(LASER_PWM_PIN, True)
        # time.sleep(duty_cycle / 1000.0)  # For 1kHz PWM
        # self.GPIO.output(LASER_PWM_PIN, False)
        # time.sleep((100 - duty_cycle) / 1000.0)
        
        # Debug output
        if abs(duty_cycle) > 0.1:  # Only print if actually doing something
            print(f"[LASER] PWM → {duty_cycle:5.1f}%")
        
        # TODO: Add thermal rollback if duty > 70% for >30 seconds


def run_control_loop():
    """
    Main control loop: Voltage → Temperature → PID → Laser PWM
    
    Loop timing target: 50Hz (20ms period)
    Current performance: ~45ms (needs optimization)
    
    Control strategy:
    1. Read all sensors (slowest part)
    2. Run neural network inference
    3. Check safety limits (CRITICAL!)
    4. Calculate PID correction
    5. Update laser PWM
    6. Repeat until thermal death or Ctrl+C
    """
    
    # ========================================================================
    # LOAD TRAINED MODEL
    # ========================================================================
    print("\n" + "="*60)
    print("RNI REAL-TIME CONTROLLER v2.3")
    print("="*60)
    
    model = RNI_Net()
    
    # Try to load weights (cross your fingers)
    checkpoint_path = "checkpoints/rni_ep150.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"[MODEL] Loaded weights from {checkpoint_path}")
        print(f"[MODEL] Total params: {sum(p.numel() for p in model.parameters()):,}")
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[ERROR] Using RANDOM weights - WILL PRODUCE GARBAGE!")
        print("[ERROR] Abort now unless you enjoy thermal runaway...")
        time.sleep(2)  # Give them time to panic
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        print("[ERROR] Model may be incompatible with this architecture")
    
    model.eval()  # Set to inference mode (disables dropout, etc.)
    
    # ========================================================================
    # INITIALIZE HARDWARE
    # ========================================================================
    hw = HardwareInterface()
    
    # PID state variables
    integral_error = 0.0  # Accumulated error
    last_error = 0.0      # For derivative term
    
    # Performance monitoring
    loop_count = 0
    max_loop_time = 0.0
    
    print("\n[CONTROL] Starting PID loop...")
    print("[CONTROL] Target: {:.1f}°C | Safety limit: {:.1f}°C".format(TARGET_TEMP, SAFETY_LIMIT))
    print("[CONTROL] Press Ctrl+C to emergency stop\n")
    
    # Small delay to let user read messages
    time.sleep(1)
    
    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================
    try:
        while True:
            loop_start = time.time()
            loop_count += 1
            
            # ----------------------------------------------------------------
            # 1. READ SENSORS (Bottleneck)
            # ----------------------------------------------------------------
            voltages = hw.read_all_sensors()
            
            # ----------------------------------------------------------------
            # 2. NEURAL NETWORK INFERENCE
            # ----------------------------------------------------------------
            with torch.no_grad():  # Disable gradient calculation for speed
                temperature_map = model(voltages)
            
            # ----------------------------------------------------------------
            # 3. SAFETY CHECKS (DO NOT MODIFY WITHOUT TESTING)
            # ----------------------------------------------------------------
            current_max_temp = temperature_map.max().item()
            avg_temp = temperature_map.mean().item()
            
            # EMERGENCY SHUTDOWN CONDITION
            if current_max_temp > SAFETY_LIMIT:
                print("\n" + "!"*60)
                print(f"EMERGENCY: Hotspot detected at {current_max_temp:.1f}°C!")
                print(f"Safety limit: {SAFETY_LIMIT}°C")
                print("Killing laser NOW")
                print("!"*60)
                hw.set_laser(0)
                
                # Additional cooldown period before allowing restart
                print("[SAFETY] Entering 5-second cooldown...")
                time.sleep(5)
                break  # Exit control loop entirely
            
            # Warning threshold (soft limit)
            if current_max_temp > SAFETY_LIMIT - 2.0:
                print(f"[WARNING] Approaching limit: {current_max_temp:.1f}°C")
            
            # ----------------------------------------------------------------
            # 4. PID CALCULATION
            # ----------------------------------------------------------------
            error = TARGET_TEMP - avg_temp
            
            # Integral term with anti-windup
            integral_error += error
            if integral_error > 50.0:   # Upper clamp
                integral_error = 50.0
            elif integral_error < -50.0:  # Lower clamp
                integral_error = -50.0
            
            # Derivative term (rate of change)
            derivative = error - last_error
            
            # PID output
            pid_output = (Kp * error) + (Ki * integral_error) + (Kd * derivative)
            
            # ----------------------------------------------------------------
            # 5. ACTUATE LASER
            # ----------------------------------------------------------------
            hw.set_laser(pid_output)
            
            # Update for next iteration
            last_error = error
            
            # ----------------------------------------------------------------
            # 6. PERFORMANCE MONITORING
            # ----------------------------------------------------------------
            loop_time = time.time() - loop_start
            max_loop_time = max(max_loop_time, loop_time)
            
            # Print status every 100 loops (about 2 seconds at 50Hz)
            if loop_count % 100 == 0:
                print(f"[STATUS] Loop {loop_count:4d} | "
                      f"Avg: {avg_temp:5.1f}°C | "
                      f"Max: {current_max_temp:5.1f}°C | "
                      f"PID: {pid_output:6.1f}% | "
                      f"Time: {loop_time*1000:4.1f}ms")
            
            # Warning for slow loops
            if loop_time > 0.05:  # 50ms threshold (20Hz)
                print(f"[LAG] Loop took {loop_time*1000:.1f}ms (>50ms target)")
            
            # ----------------------------------------------------------------
            # 7. MAINTAIN LOOP RATE
            # ----------------------------------------------------------------
            # Target: 50Hz = 20ms period
            # Sleep for remaining time
            time_to_sleep = 0.02 - loop_time  # 20ms period
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            # else: we're already running slow, just continue
            
    except KeyboardInterrupt:
        print("\n\n[CONTROL] User interrupt detected!")
        print("[CONTROL] Ramping down laser...")
    
    finally:
        # ====================================================================
        # CLEANUP (ALWAYS RUNS, EVEN ON ERROR)
        # ====================================================================
        print("\n" + "="*60)
        print("CONTROLLER SHUTDOWN")
        print("="*60)
        
        # Ensure laser is off
        hw.set_laser(0)
        print("[SAFETY] Laser: OFF")
        
        # Print performance stats
        if loop_count > 0:
            print(f"[STATS] Ran {loop_count} control loops")
            print(f"[STATS] Maximum loop time: {max_loop_time*1000:.1f}ms")
            print(f"[STATS] Target was 20ms (50Hz)")
        
        print("[SAFETY] System safe to power off")
        print("\n")


if __name__ == "__main__":
    """
    Entry point.
    
    Note: This must run as root on Raspberry Pi for GPIO access:
        sudo python3 control_loop.py
    
    Add to crontab @reboot for autonomous operation.
    """
    
    # Quick sanity check before starting
    print("[BOOT] Performing pre-flight checks...")
    
    # Check if we can import torch (sometimes breaks on RPi)
    try:
        torch.zeros(1)
    except Exception as e:
        print(f"[ERROR] PyTorch failed: {e}")
        print("[ERROR] Did you install torch for ARM?")
        exit(1)
    
    # Check for checkpoint file
    if not os.path.exists("checkpoints/rni_ep150.pth"):
        print("[WARNING] No trained model found!")
        print("[WARNING] Controller will use random weights")
        print("[WARNING] This is UNSAFE for production!")
        
        response = input("Continue anyway? (yes/NO): ")
        if response.lower() != "yes":
            print("[SAFETY] Aborting as requested")
            exit(0)
    
    # Run main loop
    run_control_loop()
