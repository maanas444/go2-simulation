#!/usr/bin/env python3
import os
import sys
import time
import struct
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# 1. --- REMOTE HANDLER CLASS ---
class UnitreeRemoteHandler:
    """
    Decodes the raw 40-byte wireless_remote array from the Unitree SDK.
    Mappings:
    - Sticks: LX, RX, RY, L2, LY (as 32-bit floats)
    - Buttons: Bitmask on Bytes 2 & 3
    """
    def __init__(self, deadzone=0.1):
        self.deadzone = deadzone
        self.state = {
            "lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0,
            "a": 0, "b": 0, "x": 0, "y": 0,
            "f1": 0, "start": 0
        }

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def update(self, remote_data):
        if not remote_data or len(remote_data) < 40:
            return self.state

        # Parse Buttons (Bitmask)
        b1, b2 = remote_data[2], remote_data[3]
        self.state["f1"] = (b1 >> 6) & 1
        self.state["start"] = (b1 >> 2) & 1
        self.state["a"] = (b2 >> 0) & 1
        self.state["b"] = (b2 >> 1) & 1
        self.state["x"] = (b2 >> 2) & 1
        self.state["y"] = (b2 >> 3) & 1

        # Parse Joysticks (Unpack 5 Floats)
        try:
            # Sticks order in SDK: LX, RX, RY, L2, LY
            sticks = struct.unpack('<5f', bytes(remote_data[4:24]))
            self.state["lx"] = self._apply_deadzone(sticks[0])
            self.state["rx"] = self._apply_deadzone(sticks[1])
            self.state["ry"] = self._apply_deadzone(sticks[2])
            self.state["ly"] = self._apply_deadzone(sticks[4])
        except struct.error:
            pass 

        return self.state

# 2. --- MAIN EXECUTION ---
def main():
    # Setup path for robot environment
    sdk_path = '/home/unitree/unitree_sdk2_python'
    if os.path.exists(sdk_path):
        sys.path.insert(0, sdk_path)

    # Robot-side Interface Setup
    # 'lo' is the local loopback for the onboard Orin
    interface = "eth0" 
    print(f"[INFO] Initializing Robot Local Channel on: {interface}")
    
    try:
        ChannelFactoryInitialize(0, interface)
    except Exception as e:
        print(f"[WARN] Failed on 'lo', trying 'eth0': {e}")
        ChannelFactoryInitialize(0, "eth0")

    handler = UnitreeRemoteHandler(deadzone=0.1)

    # Callback function to process incoming data
    def low_state_handler(msg: LowState_):
        # Decode the remote data from the message
        inputs = handler.update(msg.wireless_remote)
        
        # LOGGING: Print when sticks are moved or buttons are pressed
        if any(abs(v) > 0 for v in [inputs['lx'], inputs['ly'], inputs['rx'], inputs['ry']]) or inputs['f1']:
            print(f"Sticks -> LX: {inputs['lx']:.2f} | LY: {inputs['ly']:.2f} | RX: {inputs['rx']:.2f} | RY: {inputs['ry']:.2f}")
            if inputs['f1']: print("--- F1 Pressed ---")

    # Subscribe to the lowstate topic (rt/lowstate)
    print("[INFO] Subscribing to rt/lowstate...")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(low_state_handler, 10)

    print("[READY] Listening for Unitree Remote commands. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(0.01) # 100Hz heartbeat
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Exiting gracefully.")

if __name__ == "__main__":
    main()