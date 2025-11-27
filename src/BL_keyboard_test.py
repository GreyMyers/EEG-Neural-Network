import serial
import keyboard
import time
import serial.tools.list_ports

# --- Display available ports ---
print("Available COM ports:")
for p in serial.tools.list_ports.comports():
    print(" ", p.device)

# --- Try to open COM4 ---
try:
    ser = serial.Serial("COM5", 115200, timeout=1)
    print("\nCONNECTED to COM6 successfully!\n")
except Exception as e:
    print("\nFAILED to connect to COM6:")
    print(e)
    input("Press Enter to exit...")
    quit()

print("Controls:")
print("  W = Forward")
print("  S = Backward")
print("  A = Left")
print("  D = Right")
print("  SPACE = Stop")
print("  + Increase speed")
print("  - Decrease speed")
print("  Q = Quit\n")

last_keepalive = time.time()

# --- Main control loop ---
while True:
    time.sleep(0.03)

    # --- Send keep-alive every 2 seconds ---
    if time.time() - last_keepalive > 2:
        ser.write(b'K')
        last_keepalive = time.time()

    # --- Movement keys ---
    if keyboard.is_pressed("w"):
        ser.write(b"W")
    elif keyboard.is_pressed("s"):
        ser.write(b"S")
    elif keyboard.is_pressed("a"):
        ser.write(b"A")
    elif keyboard.is_pressed("d"):
        ser.write(b"D")
    elif keyboard.is_pressed(" "):
        ser.write(b" ")

    # --- Speed control ---
    elif keyboard.is_pressed("+"):
        ser.write(b"+")
    elif keyboard.is_pressed("-"):
        ser.write(b"-")

    # --- Quit ---
    elif keyboard.is_pressed("q"):
        print("Quitting...")
        break
