import serial
ser = serial.Serial('COM3',57600, timeout=1)  # open serial port
print(ser.name)         # check which port was really used
ser.write(b'hello')     # write a string
ser.close()             # close port