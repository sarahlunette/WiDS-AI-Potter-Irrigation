from machine import Pin, PWM
import network
import socket

# WiFi Credentials
SSID = "your_wifi"
PASSWORD = "your_password"

# Connect to WiFi
wifi = network.WLAN(network.STA_IF)
wifi.active(True)
wifi.connect(SSID, PASSWORD)

while not wifi.isconnected():
    pass

print("Connected! IP Address:", wifi.ifconfig()[0])

# PWM Setup (GPIO 2, duty cycle 0-1023)
valve_pwm = PWM(Pin(2), freq=1000)

# Function to set valve opening (1-10 scale)
def set_valve(value):
    duty = int((value / 10) * 1023)  # Convert 1-10 scale to 0-1023 PWM
    valve_pwm.duty(duty)
    return f"Valve set to {value}/10 (PWM {duty})"

# HTTP Server for receiving control commands
def web_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 80))
    s.listen(5)

    while True:
        conn, addr = s.accept()
        request = conn.recv(1024)
        request = str(request)

        # Extract value from URL (e.g., /set?value=5)
        if "/set?" in request:
            try:
                value = int(request.split("value=")[1].split()[0])
                if 1 <= value <= 10:
                    response = set_valve(value)
                else:
                    response = "Error: Value must be between 1-10"
            except:
                response = "Error: Invalid value"
        else:
            response = "Send command like /set?value=5"

        conn.send("HTTP/1.1 200 OK\n")
        conn.send("Content-Type: text/plain\n")
        conn.send("Connection: close\n\n")
        conn.send(response)
        conn.close()

# Run the server
web_server()
