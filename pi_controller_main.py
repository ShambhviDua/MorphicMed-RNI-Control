"""
RNI Real-Time Controller v2.1 - Hot-Or-Not Machine
Direct hardware interface for the resistive network laser control system.
WARNING: This actually controls 50W of laser power. Test thoroughly in sim first.

Last modified: 2025-12-16 - after PID tuning session
Hardware: Raspberry Pi 4 + Custom 308-node MUX board (Rev. C)

To run in simulation mode: python control_loop.py
To run on actual hardware: sudo python control_loop.py --real-hw
(But maybe don't until you've verified the safety logic)
"""

import time
import numpy as np
import torch
from train_rni_v3 import RNI_Net 

# ============================================================================
# PIN DEFINITIONS (BCM numbering)
# ============================================================================
LASER_PWM_PIN = 18   # PWM-capable pin for laser current control
MUX_SIG_PIN = 23     # Digital output to trigger MUX channel advance
# ADC uses SPI0_CE0 - handled by spidev if we're on real hardware

# ============================================================================
# PID CONSTANTS (TUNED 2024-03-15 AFTER 3 HOURS OF OVERSHOOT HELL)
# ============================================================================
Kp = 2.5   # "Punch" - needs to be aggressive to overcome thermal inertia
Ki = 0.8   # "Persist" - watch out for integral windup! 
Kd = 0.15  # "Damp" - stops the overshoot at target
# Tuning notes: 
# - Kp=1.0 was too sluggish, took 30s to reach target
# - Kp=3.0 caused 8°C overshoot (bad for tissue samples)
# - Ki=1.2 caused hunting around setpoint
# - Current values give ~±0.5°C steady-state error (good enough)

TARGET_TEMP = 45.0   # SETPOINT - what the biology team requested
SAFETY_LIMIT = 48.0  # HARDWARE KILL LIMIT - above this and we emergency stop
# Note: The polyimide substrate starts degrading at 50°C, so 48 is conservative


class HardwareInterface:
    """
    Talks to the actual hardware (or pretends to when running in sim).
    
    Hardware setup:
    - Raspberry Pi 4
    - 32-channel analog MUX daisy-chained (10 chips total = 320 channels)
    - 12-bit ADC (MCP3208) over SPI
    - 50W laser diode with TEC cooling (separate controller)
    """
    
    def __init__(self, simulate=True):
        """
        Initialize hardware or simulation mode.
        
        Args:
            simulate (bool): If True, fake all hardware calls (for testing on PC).
                             If False, actually import RPi.GPIO and spidev.
        """
        self.num_sensors = 308  # Fixed by our board design
        self.simulate = simulate
        
        if not simulate:
            # REAL HARDWARE MODE - uncomment when actually running on Pi
            # import RPi.GPIO as GPIO
            # import spidev
            # 
            # GPIO.setmode(GPIO.BCM)
            # GPIO.setup(LASER_PWM_PIN, GPIO.OUT)
            # GPIO.setup(MUX_SIG_PIN, GPIO.OUT)
            # 
            # self.laser_pwm = GPIO.PWM(LASER_PWM_PIN, 1000)  # 1kHz PWM
            # self.laser_pwm.start(0)
            # 
            # self.spi = spidev.SpiDev()
            # self.spi.open(0, 0)  # Bus 0, CE0
            # self.spi.max_speed_hz = 1350000  # 1.35MHz - max stable for MCP3208
            
            print("[HARDWARE] REAL MODE - Laser control ACTIVE")
            print("[WARNING] This will actually turn on a 50W laser!")
        else:
            # SIMULATION MODE (what we use 90% of the time)
            print("[HARDWARE] Simulation mode - no actual GPIO/SPI calls")
            print("[INFO] All sensor readings are random noise")
        
        # Known hardware defects (from board testing log 2024-02-28)
        self.broken_sensors = [142, 87]  # Channels with dead ADC inputs
        # Channel 142: Cold solder joint (J12)
        # Channel 87: MUX chip U4 seems to have failed
    
    def read_all_sensors(self):
        """
        Sequentially reads all 308 voltage sensors through the MUX chain.
        
        Returns:
            torch.Tensor: Shape (1, 308) with voltage readings in volts.
        
        Notes:
            - Actual hardware time: ~40ms (too slow for 100Hz target)
            - Bottleneck is SPI transfer (12 bits × 308 = 3696 bits)
            - TODO: Implement burst read or parallel ADC 
        """
        if self.simulate:
            # Generate plausible sensor data (0.5-2.5V range typical)
            # Adding some spatial correlation so it looks like real heat patterns
            base_pattern = np.sin(np.linspace(0, 4*np.pi, self.num_sensors)) * 0.5 + 1.5
            noise = np.random.normal(0, 0.1, self.num_sensors)  # 100mV noise typical
            raw = base_pattern + noise
            raw = np.clip(raw, 0.5, 2.5)  # ADC range
        else:
            # REAL HARDWARE CODE (commented out for safety)
            # raw = np.zeros(self.num_sensors)
            # for i in range(self.num_sensors):
            #     # Advance MUX (pulse MUX_SIG_PIN high then low)
            #     GPIO.output(MUX_SIG_PIN, 1)
            #     time.sleep(0.0001)  # 100µs pulse
            #     GPIO.output(MUX_SIG_PIN, 0)
            #     
            #     # Read ADC (12-bit, 0-3.3V range)
            #     adc_data = self.spi.xfer2([0x06, 0x00, 0x00])  # MCP3208 single-ended
            #     adc_value = ((adc_data[1] & 0x0F) << 8) | adc_data[2]
            #     raw[i] = adc_value * 3.3 / 4095  # Convert to volts
            #     time.sleep(0.0001)  # 100µs between reads
            raw = np.random.uniform(0.5, 2.5, self.num_sensors)  # Placeholder
        
        # HACK: Fix broken sensors by interpolating neighbors
        # This is temporary until we get the Rev. D boards back from fab
        for broken_idx in self.broken_sensors:
            if broken_idx == 0:
                raw[broken_idx] = raw[1]  # First channel: use next one
            elif broken_idx == self.num_sensors - 1:
                raw[broken_idx] = raw[broken_idx - 1]  # Last channel: use previous
            else:
                # Linear interpolation between neighbors
                raw[broken_idx] = (raw[broken_idx - 1] + raw[broken_idx + 1]) / 2
        
        # Add batch dimension for the model
        return torch.FloatTensor(raw).unsqueeze(0)
    
    def set_laser(self, duty_cycle):
        """
        Set laser PWM duty cycle (0-100%).
        
        Args:
            duty_cycle (float): 0 = off, 100 = full power
        
        Notes:
            - Laser driver has its own TEC controller (keeps diode at 25°C)
            - Minimum stable power is 5% (below that, driver shuts off)
            - We clamp at 80% max for diode lifetime reasons
        """
        # Safety clamping
        if duty_cycle < 0:
            duty_cycle = 0
        elif duty_cycle > 80:  # 80% max for diode lifetime
            duty_cycle = 80
            print("[WARNING] Laser duty clipped at 80% (diode protection)")
        
        if not self.simulate:
            # REAL HARDWARE
            # self.laser_pwm.ChangeDutyCycle(duty_cycle)
            pass
        else:
            # Just print in simulation
            if duty_cycle > 0:
                print(f"[SIM] Laser → {duty_cycle:5.1f}% power")
            else:
                print(f"[SIM] Laser → OFF")
        
        return duty_cycle


def run_control_loop():
    """
    Main control loop: reads sensors → runs RNI inference → PID → adjusts laser.
    
    Loop target: 100Hz (10ms period)
    Actual: ~45Hz on Pi 4 (sensor read is bottleneck)
    
    Exit conditions:
    1. Ctrl+C (user interrupt)
    2. Emergency temperature limit exceeded
    3. TODO: Add watchdog timeout
    """
    
    print("\n" + "="*60)
    print("RNI REAL-TIME CONTROL LOOP v2.1")
    print("="*60)
    
    # ========================================================================
    # LOAD THE MODEL (magic neural network)
    # ========================================================================
    print("[MODEL] Loading RNI weights...")
    model = RNI_Net()
    
    # Try to load weights
    weights_path = "checkpoints/rni_ep150.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"[MODEL] Loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find weights at {weights_path}")
        print("[ERROR] Using random weights - THIS WILL NOT WORK PROPERLY!")
        print("[ERROR] Run training first or check the path.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        print("[ERROR] Model structure mismatch? Check train_rni_v3.py")
    
    model.eval()  # Set to inference mode (no dropout, etc.)
    
    # ========================================================================
    # INITIALIZE HARDWARE (in simulation mode for safety)
    # ========================================================================
    hw = HardwareInterface(simulate=True)  # CHANGE TO False ON REAL PI
    print(f"[CONTROL] Target temperature: {TARGET_TEMP}°C")
    print(f"[CONTROL] Safety limit: {SAFETY_LIMIT}°C")
    print(f"[CONTROL] PID: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    
    # ========================================================================
    # PID STATE VARIABLES
    # ========================================================================
    integral_error = 0.0  # Accumulated error over time
    last_error = 0.0      # Previous error (for derivative)
    loop_count = 0        # For occasional status printouts
    
    # Anti-windup limits (prevent integral from getting too big)
    INTEGRAL_MAX = 50.0
    INTEGRAL_MIN = -50.0
    
    print("\n[CONTROL] Entering main loop. Press Ctrl+C to abort.")
    print("[CONTROL] Starting in 3 seconds...")
    time.sleep(3)  # Safety delay
    
    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================
    try:
        while True:
            loop_start = time.time()
            loop_count += 1
            
            # --- STEP 1: READ SENSORS ---
            # This is the slow part (~40ms on hardware)
            voltages = hw.read_all_sensors()
            
            # --- STEP 2: RUN RNI INFERENCE ---
            # Convert voltages to temperature map (our neural network magic)
            with torch.no_grad():  # Don't compute gradients (faster)
                temp_map = model(voltages)
            
            # --- STEP 3: EXTRACT METRICS ---
            # temp_map shape: (1, 308) - one temperature per sensor node
            current_max_temp = temp_map.max().item()
            current_avg_temp = temp_map.mean().item()
            current_min_temp = temp_map.min().item()
            
            # --- STEP 4: SAFETY CHECK (NON-NEGOTIABLE) ---
            if current_max_temp > SAFETY_LIMIT:
                print(f"\n[EMERGENCY SHUTDOWN] Hotspot detected: {current_max_temp:.1f}°C!")
                print(f"[EMERGENCY] Safety limit is {SAFETY_LIMIT}°C")
                print(f"[EMERGENCY] Killing laser immediately!")
                hw.set_laser(0)
                print(f"[EMERGENCY] Loop terminated.")
                break  # Exit the control loop
            
            # --- STEP 5: PID CALCULATION ---
            # Error = difference between target and average temperature
            error = TARGET_TEMP - current_avg_temp
            
            # Integral term (accumulated error)
            integral_error += error
            
            # Anti-windup: clamp integral term
            if integral_error > INTEGRAL_MAX:
                integral_error = INTEGRAL_MAX
                # print("[PID] Integral clamped at MAX")  # Too verbose
            elif integral_error < INTEGRAL_MIN:
                integral_error = INTEGRAL_MIN
                # print("[PID] Integral clamped at MIN")
            
            # Derivative term (rate of change)
            derivative = error - last_error
            
            # PID output formula
            pid_output = (Kp * error) + (Ki * integral_error) + (Kd * derivative)
            
            # Save error for next derivative calculation
            last_error = error
            
            # --- STEP 6: ACTUATE LASER ---
            # Convert PID output (temperature error) to laser duty cycle
            # PID output is in °C, we need 0-100% duty
            # Rough calibration: 1°C error ≈ 2% duty cycle change
            base_duty = 50.0  # 50% duty gives roughly TARGET_TEMP at steady state
            duty_cycle = base_duty + pid_output * 2.0
            
            # Actually set the laser (or print in sim mode)
            actual_duty = hw.set_laser(duty_cycle)
            
            # --- STEP 7: STATUS REPORTING (every 100 loops = ~2 seconds) ---
            if loop_count % 100 == 0:
                print(f"[STATUS] Loop {loop_count}: "
                      f"Max={current_max_temp:5.1f}°C, "
                      f"Avg={current_avg_temp:5.1f}°C, "
                      f"Min={current_min_temp:5.1f}°C, "
                      f"Laser={actual_duty:5.1f}%, "
                      f"Error={error:5.2f}°C")
            
            # --- STEP 8: LOOP TIMING ---
            # Target loop period: 10ms (100Hz)
            loop_time = time.time() - loop_start
            
            if loop_time > 0.05:  # 50ms threshold
                print(f"[TIMING WARNING] Loop took {loop_time*1000:.1f}ms "
                      f"(target: 10ms)")
            
            # Sleep to maintain roughly 100Hz rate
            # In reality, the sensor read (40ms) is the bottleneck
            time_to_sleep = max(0, 0.01 - loop_time)  # Aim for 10ms total
            time.sleep(time_to_sleep)
    
    except KeyboardInterrupt:
        print("\n\n[CONTROL] User interrupt (Ctrl+C) detected.")
        print("[CONTROL] Shutting down safely...")
    
    finally:
        # ALWAYS ensure laser is off when we exit
        print("[SAFETY] Turning laser OFF")
        hw.set_laser(0)
        print("[SAFETY] Shutdown complete.")
        
        if loop_count > 0:
            avg_loop_time = (time.time() - loop_start) / loop_count * 1000
            print(f"[STATS] Ran {loop_count} loops, average {avg_loop_time:.1f}ms/loop")


if __name__ == "__main__":
    # Entry point with some basic argument parsing
    import sys
    
    print("\n" + "="*60)
    print("RNI LASER CONTROL SYSTEM")
    print("="*60)
    print("")
    print("WARNING: This software controls high-power laser equipment.")
    print("         Ensure all safety protocols are followed.")
    print("         Authorized personnel only.")
    print("")
    
    # Check for real hardware flag
    if "--real-hw" in sys.argv:
        print("[WARNING] REAL HARDWARE MODE requested.")
        print("[WARNING] This will actually control the laser!")
        print("[WARNING] Type 'YES' to continue: ", end="")
        response = input()
        if response == "YES":
            print("[INFO] Proceeding with real hardware control...")
            # You'd need to modify HardwareInterface(simulate=False)
            print("[ERROR] Real hardware mode not implemented in this version.")
            print("[ERROR] Edit HardwareInterface class to enable.")
        else:
            print("[INFO] Aborting. Running in simulation mode.")
    
    print("\nStarting control system (simulation mode)...")
    run_control_loop()
