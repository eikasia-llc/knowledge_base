# **ARDUINO AGENT ARCHITECTURE**

**Status:** Draft

**Version:** 1.0.0

**Context:** Hardware-in-the-Loop (HIL) Integration / Multi-Agent Systems

## **1\. Overview**

The **Arduino Agent** represents the physical interface layer of the system. Unlike purely software-based agents that manipulate digital information, this agent is responsible for **actuation** (affecting the physical world) and **sensing** (digitizing physical states).

In the context of the Model Context Protocol (MCP), the Arduino Agent does not "think" in natural language. Instead, it operates as a deterministic **"Edge Executor"** that exposes its atomic capabilities (GPIO control, PWM signals, sensor reads) as high-level Tools to the central orchestration layer.

## **2\. Core Concepts**

### **2.1 The Hierarchical Control Model**

The integration follows a strict command hierarchy to ensure stability in the simulation:

* **The Planner (LLM):** Operates in the semantic space. Determines *intent* (e.g., "Cool down the system").  
* **The Bridge (MCP Server):** Operates in the logic space. Translates intent into specific function calls and validates safety limits.  
* **The Controller (Arduino):** Operates in the signal space. Executes the voltage changes or reads raw analog values.

### **2.2 State Space (![][image1])**

The agent's state is defined by the instantaneous values of its connected peripherals. Unlike the LLM's context window, this state is volatile and must be polled.

### **![][image2]2.3 Action Space (![][image3])**

The agent exposes a discrete set of deterministic actions. These are mapped 1:1 to MCP Tools.

## **![][image4]3\. Infrastructure Architecture**

The architecture relies on a **"Host-Satellite"** topology. The Arduino is the Satellite, and the machine running the MCP Server is the Host.

\[ Orchestrator \] \<===\> \[ MCP Server (Host) \] \<===\> \[ Serial Bus \] \<===\> \[ Arduino (Satellite) \]  
      (AI)                 (Python/Node)             (USB/UART)             (C++/Firmware)

### **3.1 The Communication Pipeline**

1. **Tool Call:** Orchestrator requests move\_servo(angle=90).  
2. **Validation:** MCP Server checks if ![][image5].  
3. **Serialization:** MCP Server converts request to bytecode (e.g., M:90\\n).  
4. **Execution:** Arduino parses bytecode and triggers hardware interrupt.  
5. **Feedback:** Arduino returns success/failure signal.

## **4\. Implementation Plan**

### **Phase 1: Firmware Layer (C++)**

* Develop a non-blocking SerialListener loop.  
* Implement a command parser (e.g., splitting strings by delimiters).  
* Define "Emergency Stop" hard-coded logic for safety.

### **Phase 2: Driver Layer (Python)**

* Implement pyserial connection management with auto-reconnect.  
* Create a DeviceController class to abstract raw serial commands.  
* **Research Note:** Ensure baud rate matches hardware capabilities (recommend 115200 for low latency).

### **Phase 3: Agent Layer (MCP)**

* Wrap DeviceController methods with @mcp.tool decorators.  
* Add docstrings optimized for LLM comprehension.  
* Implement "State Caching" to reduce hardware polling frequency.

## **5\. Security & Safety**

* **Hardware Limits:** Firmware must cap PWM values regardless of software commands.  
* **Watchdog Timers:** Arduino should reset if no heartbeat is received from the MCP Server for ![][image6].  
* **Sanitization:** The Bridge must reject non-ASCII or malformed payloads before transmission.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAABO0lEQVR4Xu2SvStGURzHz4O8hLLcbum+1tWtK6nHIMliEmVkUzI9g8HiHzBSkvgLLHabnk1ZhMUkWZXByOLl87udR+f8MlqUb30793xffuc+57nG/Fk04jieS9O0mWVZvwjsR6MoGtDBGgQ3YBsewjP4RHGV9b4sy2GdN0mSbGNeEhrpaEwfQ/uAV262BoVxjHdCE9pDv4B7Wu+c8hkEwZD27KsuaV2MIynBHbZdyvu+EA/czrItCZ8JncL1oij6dNZFg9AWhVenLDzH69FhD2EYDhJc5Dfus75JkWELOmfyPJ/UmoBwy5ZankEhxGh7ogX6jJQ4dcozEFYwHnjs9gxTn3SA92jUbcq0YzttzdXZz6K/wKar10C8JrDJegdv4C48gbc/XoAAY1rWqqp65cuW/4bCvM794zfwBevtS6FR+ne9AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAABBklEQVR4XmNgGL5ATk5OUEFBoRJdnCggLy8/DYj/AJlM6HJ4AdBGA6DGv0D8X0VFRRRdHi8AatoPxHdAmoHO10aXxwmAtkYC8USgxlkgzUDsiK4GKxAXF+cGKj4H1CwApFtAmoHscHR1WAFQcSsQJ4PYQE0FUGfnoavDALKysspADYeATEYQH6gpFursFjSlmACoeIOioqIZjA/U5AXVPAtZHQrQ0tJiAyo4A7RVAllcWlpaBqr5BLI4CgClIqCCEiziHFDN99HlQIAJKBEMxF/U1dV50SWNjY1ZgXLfQRikFi4BFLAE4tdQyV9A/AqYkviQ5EFJ8w1U/gdILTBM2uAGjIKhAADK3z+vrNTRGgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG4AAAAXCAYAAADqdnryAAAE0ElEQVR4Xu2Ye4hVVRjFb44VRZRIQzaPu+88ejiVhVPgSO+MUDEazB70sv+MhEjTHlJkZdAfYURkZhhJpj3sD3tZ1lChVtJDCjQyyKEiTQutJCLRfuueve987rlzmzvVeIyz4OOcs7619177cR775HIZMmTIkCHD/x9D8vn8JOfcvEKhcCvH42PBYKKpqQkLbgbxMOfnxnmPVHkOwMt5ipgPaGhoOA3fs/F7c3Nzcz7OC+SbyU/n+CDHiXG+iLa2tsNIvkS8pkHylX5PdMTagaCxsbGFjiykvifjXDmgvQztD8QsDQDHD/E032rS5pn2L/b6b4h98h5rBDQ3kVtNTCbGc71ei85q4C8guqnzFo4XEl1onreaIiCnkdxWV1d3ZOC4nkds5nSokVYFBvR06lhGvIKJs+N8OfiV9ovaDxyr8kSXDMaEwKXJs4B+MvorOF7jvfaaON+PjZweEjjuvuFwO5XzVA3X3xG3RZpf6fMNgSsC8lPIlZbDxDgZqMZ8gMqoPo7LOZ4R5ysB/b2+452W5/onYpm5To1nC9rv8P57TRx1Xwu/o7W19WjLw20gN0Xn3OmX+vKjI80a4u0SgclhEnJ8xugkHO0ruNvylYB2AvEO8RSr5IQ43x+4ZLWr3fERv4XYrvO0ebZwFSbO381aWKtoq95zJ8H9GCaTPj3iyzfZslpUcH/kwtNEZiUksdAKqfwU38gCy5dBDWWvRLsW7fz6+vqGWFAN5MMbj++47eLb29sPTZtnC1dh4gT4Lp/fhYc7OX7MXXaOyS/1+f0+tLh+UTwTfVwgig3FnYUbKb5Q7qVogGYOsYPGz4pzAwE+rvft3hg4dYLYG4ynzbNF8Ob6mDjurMPJfeY18rrembuLPr0Z+mnLqU++zMhAjBFBgSesUAIvfM7y5UAd56N7nVhCPW1xvkoMoZ511PMe5zUi8skn8W/EXnU8hZ5LcD2LanacyyV9e5zccuJyzr/1frvDRMGvEoe/EbZgmLjSRwyPnVZfeJEVshpP9fyjlq8Eypzpklt6xT9ZzbW1tUdRx2LMfk68wHknsY3YqnwaPQe4nom7vUxuJrE2578q9V5TP6Uv+Mc+58/qOrwDTVltfcQPLxJ+kPZoRiPh2L4M/B2o62Rv6A3KXxTnBwCt1D3ECl2k2bOrMHG08RX8dWX497VI/flDKu/CI7FH8xbc7pzZSqixd4kPemRF4VWqgFt4lOWrQUtLS2Mh+UrqwvCknG20D/hH4UpiauAKyWNNg1Ea0DR5tggTRx13lMnps7/XxOWTjfZqnYe+crzEauC+JF62nMT6wtrNo6IucFwv0eAY2YDBJvlY6rpPAxLnYqAZgfZPYqmueaYfIx8uem+lybNFIfnTs8+ZHwgBTNA9xEc87o4InM7RrlF/PKUN+Be2Xb9x/52y4wJXAon7iU1+9vWcXRdvFAcLtL0YH0+75F/lZpdMWq+/IWnyzEDP9V61bdnlkr8/3cSrRqZ/qwvgfiYeIx4gPiHuMhpNVN4l+1Y9ujUGX1P/1VazH7Ta88lvmw7tl+L8YEJ7NXmRpzhnkSbP/YU+PFzywdVZYQ85VH2ib1Pcf/HznIEbpkHuT8R7kwOFg9Hzvw6/Ihb1M2bG5Q8EDkbPGTJkyJChavwFWO/mn85TQg0AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAYCAYAAAC1Ft6mAAAC00lEQVR4Xu2WSWhTURSG0ziCA1qNsZleYiLR4EAJCiqiC3FYuKriQhdKRcG1goLjog5VFNyoqFkoLkRBVw64EK02DogKbrooLgRREHTlwoLxO3nvhZuTpr7Xiovyfvi574z33HvPvUkoFCBAgFGBdDq9Gu7IZDKLGScyLrIsq5PvVdrXM1Kp1BqSXNP6/wEKP8bcFcVnsE37egbBz2FZ6wdBOJFItGrlSMCCjjL3B/ga3oB7isXiOO3nGZFIZDJJBkh8Uts0crncVHxvw8u0Rl7bhwO64zBzb9d635Di2O0cCbfJMZN0N/JcL7uDfzv+N2VxLGyptvsBOQ75WVA8Hp+B/zrilrOGCTWDHC2Lecj4WU5IvoUsKm7EDwk5JWJL8L7cQ233AmIPUuApxluwB97LZrNJ7Sdgjo3Y3+N/gPEMfNxwAJZ9f3rrlD4hBZDjPBM9ZexAFdY+zeAU9yYWi80UmaI3If9Av177ou+T03G+O+Bv5p5Vc4hGo5NQ/iLJiZpyBKAFIuTrkg2SVkY1RvtoUOBsOM3UEf8R9oeMjUGeb9kv4FW6aCEx8+TEjLBqy6wVJ3fV/wKySWn7Kf4kvynaPghatMKyu6ZCwXNMPXlfOYsSluVBM+1yvMcxDDQYhgF5zsl3hHwv4M5CoTBe+2gkk8kYvl+JO2fqndatYF9g6p3W3of/E2dRV0y77IS0xktDLjGMNVz+CmLa4GnYQyFbQh7azIXzqDQUhtwHf7oXnu+t8JvxUxFGfkftF42wquMXNxnGXXzvrXMYAtIOkpCYR3CDtnsFOR7wFCcMuSiLlMfC1SHfgW/lp0ZkeUCQ+znBJa5PFQTtx/Cd8RJjV52xCSz7cl5n4rtwhbb7BUWtJF8ZnoXdziZ3h4xOQW6HvbDEnBdkE6xmmygvk9c7lM/np1j2vwQvl90PWsi7jEI3s7lpbXQhJ4nPdK0PECBAgNGPP7jXslINKgZPAAAAAElFTkSuQmCC>