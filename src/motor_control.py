"""
Motor Control Module for ESP32 via Bluetooth
Controls motors based on BCI predictions (left/right actions)
"""

import serial
import serial.tools.list_ports
import time
import threading


class MotorController:
    def __init__(self, device_name="ESP32_MCPWM_Motor", baudrate=115200, auto_connect=True):
        """
        Initialize motor controller for ESP32 Bluetooth communication
        
        Args:
            device_name: Bluetooth device name to search for
            baudrate: Serial baudrate (default 115200)
            auto_connect: Automatically connect on initialization
        """
        self.device_name = device_name
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        self.lock = threading.Lock()
        self.last_command = None
        self.command_timeout = 0.5  # seconds between same commands
        self.last_command_time = 0
        
        if auto_connect:
            self.connect()
    
    def find_bluetooth_port(self):
        """
        Find the COM port for the ESP32 Bluetooth device
        
        Returns:
            str: COM port name or None if not found
        """
        ports = serial.tools.list_ports.comports()
        
        # Try to find ESP32 by description or device name
        for port in ports:
            port_desc = port.description.lower()
            if 'esp32' in port_desc or 'bluetooth' in port_desc or 'standard serial' in port_desc:
                print(f"Found potential ESP32 port: {port.device} - {port.description}")
                return port.device
        
        # If not found, list all available ports
        print("\nAvailable COM ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
        
        return None
    
    def connect(self, port=None, timeout=5):
        """
        Connect to ESP32 via Bluetooth/Serial
        
        Args:
            port: Specific COM port to use (auto-detect if None)
            timeout: Connection timeout in seconds
        
        Returns:
            bool: True if connected successfully
        """
        try:
            if port is None:
                port = self.find_bluetooth_port()
            
            if port is None:
                print("ERROR: Could not find ESP32 Bluetooth device")
                print("Please pair your ESP32 via Windows Bluetooth settings first")
                return False
            
            print(f"Connecting to {port}...")
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=timeout
            )
            
            time.sleep(2)  # Wait for connection to stabilize
            
            # Test connection by stopping motors
            self.send_command('X')
            
            self.connected = True
            print(f"Successfully connected to ESP32 on {port}")
            return True
            
        except serial.SerialException as e:
            print(f"ERROR: Failed to connect to ESP32: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.serial_connection and self.serial_connection.is_open:
            self.stop()  # Stop motors before disconnecting
            self.serial_connection.close()
            self.connected = False
            print("Disconnected from ESP32")
    
    def send_command(self, command):
        """
        Send a single character command to ESP32
        
        Args:
            command: Single character command ('W', 'S', 'A', 'D', 'X')
        
        Returns:
            bool: True if sent successfully
        """
        if not self.connected or not self.serial_connection:
            print("ERROR: Not connected to ESP32")
            return False
        
        with self.lock:
            try:
                # Convert to uppercase and send
                cmd = command.upper()
                self.serial_connection.write(cmd.encode())
                
                # Read response from ESP32 (optional)
                time.sleep(0.05)
                if self.serial_connection.in_waiting > 0:
                    response = self.serial_connection.readline().decode().strip()
                    print(f"ESP32: {response}")
                
                return True
                
            except Exception as e:
                print(f"ERROR sending command '{command}': {e}")
                return False
    
    def send_command_throttled(self, command):
        """
        Send command with throttling to avoid flooding ESP32
        Only sends if different from last command or timeout has passed
        
        Args:
            command: Single character command
        
        Returns:
            bool: True if sent
        """
        current_time = time.time()
        
        # Send if command changed or timeout passed
        if (command != self.last_command or 
            current_time - self.last_command_time > self.command_timeout):
            
            success = self.send_command(command)
            if success:
                self.last_command = command
                self.last_command_time = current_time
            return success
        
        return False
    
    # ===== Motor Control Functions =====
    
    def forward(self):
        """Move motors forward"""
        return self.send_command('W')
    
    def backward(self):
        """Move motors backward"""
        return self.send_command('S')
    
    def left(self):
        """Turn left"""
        return self.send_command('A')
    
    def right(self):
        """Turn right"""
        return self.send_command('D')
    
    def stop(self):
        """Stop all motors"""
        return self.send_command('X')
    
    # ===== BCI Integration Functions =====
    
    def execute_action(self, action_name, throttled=True):
        """
        Execute motor action based on BCI prediction
        
        Args:
            action_name: Action string ('left', 'right', 'forward', 'backward', 'stop')
            throttled: Use throttled sending to avoid flooding
        
        Returns:
            bool: True if executed successfully
        """
        action_map = {
            'left': 'A',
            'right': 'D',
            'forward': 'W',
            'stop': 'X',
        }
        
        action_lower = action_name.lower()
        if action_lower not in action_map:
            print(f"WARNING: Unknown action '{action_name}'")
            return False
        
        command = action_map[action_lower]
        
        if throttled:
            return self.send_command_throttled(command)
        else:
            return self.send_command(command)


# ===== Standalone Test Function =====

def test_motor_controller():
    """Test motor controller with keyboard input"""
    print("=" * 50)
    print("Motor Controller Test")
    print("=" * 50)
    
    # Create controller
    motor = MotorController()
    
    if not motor.connected:
        print("\nPlease specify COM port manually:")
        port = input("Enter COM port (e.g., COM5): ").strip()
        motor.connect(port=port)
    
    if not motor.connected:
        print("Failed to connect. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("Controls:")
    print("  W = Forward")
    print("  S = Backward")
    print("  A = Left")
    print("  D = Right")
    print("  X = Stop")
    print("  Q = Quit")
    print("=" * 50)
    
    try:
        while True:
            cmd = input("\nEnter command: ").strip().upper()
            
            if cmd == 'Q':
                break
            elif cmd in ['W', 'S', 'A', 'D', 'X']:
                motor.send_command(cmd)
            else:
                print("Invalid command")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        motor.disconnect()
        print("Test complete")


if __name__ == '__main__':
    test_motor_controller()