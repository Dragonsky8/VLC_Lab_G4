import os
from time import sleep
from adb_shell.auth.keygen import keygen
from adb_shell.adb_device import AdbDeviceTcp, AdbDeviceUsb
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
from adb_shell.exceptions import *

def copyVideo(vidfile = 'video01', DSLRdir = 'sdcard/DCIM/OpenCamera/'):

    #Destination file
    #vidfile = 'video01'
    #DSLRdir = 'sdcard/DCIM/OpenCamera/'

    #List files in the directory
    fs = os.listdir()
    #Key files
    adbkey = os.path.join(os.getcwd(),'adbkey')

    #Generate keys only if the first time
    if not('adbkey' in fs and 'adbkey.pub' in fs):
        keygen(adbkey)

    # Load the public and private keys
    with open(adbkey) as f:
        priv = f.read()
    with open(adbkey + '.pub') as f:
        pub = f.read()
    signer = PythonRSASigner(pub, priv)

    # Connect
    #device1 = AdbDeviceTcp('192.168.0.222', 5555, default_transport_timeout_s=9.)
    #device1.connect(rsa_keys=[signer], auth_timeout_s=0.1)

    #We loop until the device is connected but other conditions are met.
    connected = False

    while not(connected):
        try:
            device1 = AdbDeviceUsb()
            connected = device1.connect(rsa_keys=[signer], auth_timeout_s=0.1)
        except InvalidTransportError:
            print('Install usb extra options: pip install adb-shell[usb]')
            #This exception needs to exit the loop until the module is installed and reloaded.
            connected = None
            break
        except UsbDeviceNotFoundError:
            print('Device not found, connect device via USB with USB debugging enabled.')
            sleep(5)
            continue
        except UsbReadFailedError:
            print('Maybe need to accept RSA from the phone. Otherwise, try to replug the usb.')
            sleep(5)
            continue
        except:
            print('USB error. Probably timeout. Try unplug and replug the device.')
            device1.close()
            sleep(5)
            continue

    #General case where the pip install adb-shell[usb] is required.
    if connected == None:
        vidfile = ''
    else:
        # Get files in DLSRCamera
        timeout = True
        while(timeout):
            try:
                dslrDir = device1.list(DSLRdir)
                timeout = False
            except:
                print('Be patient...')
                device1.close()
                device1.connect(rsa_keys=[signer], auth_timeout_s=0.1)
        dslrFiles = []
        #Create a list of files
        for f in dslrDir:
            dslrFiles.append(f.filename.decode())
        #File names are saved in date format, thus reverse sorting. This also gets the video files
        #because they start with 'VID' and images with 'IMG'.
        #TODO: add filter for video files if necessary.
        dslrFiles.sort(reverse=True)
        #If directory is incorrect = empty list
        if len(dslrFiles) == 0:
            print('Check your camera directory. No files found.')
            vidfile = ''
        else:
            #Get the extension from the origin file
            vidfile = vidfile + dslrFiles[0][-4:]
            vidpath = os.path.join(os.getcwd(), vidfile)
            response = device1.pull(DSLRdir + dslrFiles[0], vidpath)
        device1.close()

    return vidfile
