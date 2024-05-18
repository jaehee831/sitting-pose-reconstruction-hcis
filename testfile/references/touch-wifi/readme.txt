Hey, 
   The board should get there, I hope. 

About the power:
    The board is power through either 5V or 6 -14V using VoltagePin. 
   Pin1 :  6-14V 
    Pin2:   GND    
   Pin3 :  5V
     I plug in power from a USB serial to power the board, so right now, if you just want to power the board, you can just plug in the USB.  
    If you plan to operate using a battery, I suggest finding a batter with voltage greater than 6 voltage and use another pin.

Testing:
     Connect your computer to SSID:   PressureArray    password is PressureArray
     You should get IP address as 192.168.4.2  If there is more than one computer connect to, you might have to disconnect another one and set your IPto 192.168.4.2. 

     I put all my code in main.zip attached here.    If you unzip the file, there is a program call 
      station/main/visualize.py  
           You need to install opencv-python and numpy.  That's it.  When you run, you should see read out from ADC pressure.
     The board has a programmable potentiometer (https://th.mouser.com/datasheet/2/609/AD5241_5242-1501681.pdf)  .     You can set the values of the potentiometer by typing two number, for example
123 245
then press enter.
    Valid values are from 0 to 255.    
        
Programming:
     You can program using toolchain here: 
 https://docs.espressif.com/projects/esp-idf/en/latest/get-started/  
    If you follow through, then you should be able to compile the project I attached.
     If you want to flash a new firmware, do the following:
      Connect the USB serial to the board like the way I shipped the board.  
       there is a jumper I left at Boot Opt pin.  Short the pin, power off and power on. 
     Type 
     make flash
      It should start programming. Once done, remove the jumper and power cycle the board again.   
       You can do 
    make monitor 
   to read debug information from serial. 



   Yam


=====================

Petr Note: Disable Windows Firewall!!!!

=====================

Hi,
     The ADC is an 18-bits (https://www.ti.com/lit/ds/symlink/ads8885.pdf).   It is a differential ADC, so it will return negative output when AINP is less than AINN.  I do not have code in front of me. I remember trying to handle the negative value, but cannot remember if I gave up handling on the MCU and thought of implementing it on the host side or not. If you do see values over 500K, I believe it's when the output from ADC is negative and somehow overflow.
    Since OpenCV cannot visualize 18-bits values, I divide by 1024 so that the value is somewhat in between 0 to 255.  

I tested the board by using the visualize.py. Basically, I just pressed the pressure sensor and see the changes in the value, and I just manually ( somewhat randomly ) changed the resistor values, which changes the gain until I could see the difference. The output from the board is totally random, and (should) might be oscillating if you do not have any load.  
    I can call to talk tomorrow, just let me know the time. 
    Yam
          
		  


