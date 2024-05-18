import socket
import time
import pylsl

# SELECT DATA TO STREAM
acc = True      # 3-axis acceleration
bvp = True      # Blood Volume Pulse
gsr = True      # Galvanic Skin Response (Electrodermal Activity)
tmp = True      # Temperature

serverAddress = '172.27.182.25' # '172.27.182.18' # '127.0.0.8'
serverPort = 30000
bufferSize = 4096

deviceID = 'D931CD' # 'A02088' # A027EF

def connect():
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)

    print("Connecting to server")
    s.connect((serverAddress, serverPort))
    print("Connected to server\n")

    print("Devices available:")
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Connecting to device")
    s.send(("device_connect " + deviceID + "\r\n").encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Pausing data receiving")
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

connect()

time.sleep(1)

def suscribe_to_data():
    if acc:
        print("Suscribing to ACC")
        s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if bvp:
        print("Suscribing to BVP")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if gsr:
        print("Suscribing to GSR")
        s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if tmp:
        print("Suscribing to Temp")
        s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    print("Resuming data receiving")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))
suscribe_to_data()

def prepare_LSL_streaming():
    print("Starting LSL streaming")

    infoACC = pylsl.StreamInfo('acc', 'Acc', 3, 32, 'int32', 'E4_Acc')
    global outletACC
    outletACC = pylsl.StreamOutlet(infoACC)

prepare_LSL_streaming()

time.sleep(1)

# def reconnect():
#     print("Reconnecting...")
#     connect()
#     suscribe_to_data()
#     stream()
#
# def stream():
#     try:
#         print("Streaming...")
#         while True:
#             try:
#                 response = s.recv(bufferSize).decode("utf-8")
#                 print(response)
#                 if "connection lost to device" in response:
#                     print(response.decode("utf-8"))
#                     reconnect()
#                     break
#                 samples = response.split("\n")
#                 for i in range(len(samples)-1):
#                     stream_type = samples[i].split()[0]
#                     if stream_type == "E4_Acc":
#                         timestamp = float(samples[i].split()[1].replace(',','.'))
#                         data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
#                         outletACC.push_sample(data, timestamp=timestamp)
#                     if stream_type == "E4_Bvp":
#                         timestamp = float(samples[i].split()[1].replace(',','.'))
#                         data = float(samples[i].split()[2].replace(',','.'))
#                         outletBVP.push_sample([data], timestamp=timestamp)
#                     if stream_type == "E4_Gsr":
#                         timestamp = float(samples[i].split()[1].replace(',','.'))
#                         data = float(samples[i].split()[2].replace(',','.'))
#                         outletGSR.push_sample([data], timestamp=timestamp)
#                     if stream_type == "E4_Temperature":
#                         timestamp = float(samples[i].split()[1].replace(',','.'))
#                         data = float(samples[i].split()[2].replace(',','.'))
#                         outletTemp.push_sample([data], timestamp=timestamp)
#                 time.sleep(1)
#             except socket.timeout:
#                 print("Socket timeout")
#                 reconnect()
#                 break
#     except KeyboardInterrupt:
#         print("Disconnecting from device")
#         s.send("device_disconnect\r\n".encode())
#         s.close()
# stream()