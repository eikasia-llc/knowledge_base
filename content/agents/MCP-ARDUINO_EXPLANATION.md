# **MCP-ARDUINO PROTOCOL EXPLANATION**

**Type:** Technical Reference

**Scope:** Serial/MCP Bridge Implementation

## **1\. Introduction**

This document details the mechanics of bridging a high-level **Model Context Protocol (MCP)** server with low-level **Arduino Serial communication**. The challenge lies in bridging two fundamentally different time domains: the asynchronous, token-based generation of Large Language Models (LLMs) and the synchronous, real-time interrupt cycles of a microcontroller.

## **2\. The Translation Layer**

The MCP Server acts as a **Transducer**, converting Semantic Intents into Electrical Signals.

### **2.1 Data Flow Analysis**

The total latency (![][image1]) for a single interaction is the sum of four distinct delays:

![][image2]Where:

* ![][image3]: Time for LLM to generate the tool call (variable).  
* ![][image4]: Network latency for MCP JSON-RPC transport.  
* ![][image5]: Time to serialize Python objects to bytes.  
* ![][image6]: Hardware rise time and execution.

**Optimization Strategy:** We cannot control ![][image3], so we optimize ![][image5] by using efficient byte-packing and ![][image6] by using non-blocking Arduino code.

## **3\. Protocol Specification**

To ensure reliability, we utilize a **Command-Response** handshake protocol over UART.

### **3.1 Packet Structure (Host ![][image7] Arduino)**

Commands are sent as ASCII strings terminated by a newline character \\n. This allows the Arduino readStringUntil() function to easily parse the buffer.

**Format:** \[CMD\]:\[PAYLOAD\]\\n

| Component | Type | Example | Description |
| :---- | :---- | :---- | :---- |
| CMD | Char\[3\] | LED | 3-letter operation code. |
| : | Delimiter | : | Separator. |
| PAYLOAD | String/Int | 255 | The argument for the operation. |
| \\n | Terminator | \\n | End of packet marker. |

### **3.2 Response Structure (Arduino ![][image7] Host)**

Every command must receive an acknowledgment to close the loop.

**Format:** \[STATUS\]:\[DATA\]\\n

* **Success:** OK:1 (Operation complete)  
* **Data Return:** RET:24.5 (e.g., Temperature sensor reading)  
* **Error:** ERR:INVALID\_PIN

## **4\. Technical Challenges & Solutions**

### **4.1 The "Stale Buffer" Problem**

**Issue:** If the LLM generates commands faster than the baud rate allows, the Arduino's 64-byte serial buffer will overflow, causing dropped characters and corrupted commands.

**Solution:**

1. **Flush Input:** The Python bridge should call serial.reset\_input\_buffer() before critical reads.  
2. **Synchronous Blocking:** The MCP tool function must *block* until it receives the OK or ERR response from the Arduino before returning the result to the LLM.

### **4.2 Data Type Mismatch**

**Issue:** LLMs output floating point numbers or descriptive text (e.g., "Turn it to 50%"), while Arduino analogWrite requires 8-bit integers (![][image8]).

**Solution:**

The Python Bridge handles the normalization logic:

![][image9]This prevents the firmware from needing complex string parsing logic.

## **5\. Code Example: The Handshake Loop**

Below is the Python implementation of the atomic command execution that guarantees synchronization.

def send\_command(command: str, value: str) \-\> str:  
    """  
    Sends a command and waits for a specific ACK pattern.  
    Raises TimeoutError if hardware hangs.  
    """  
    full\_payload \= f"{command}:{value}\\n"  
      
    \# 1\. Clear any old data  
    arduino\_serial.reset\_input\_buffer()  
      
    \# 2\. Transmit  
    arduino\_serial.write(full\_payload.encode('utf-8'))  
      
    \# 3\. Wait for response (Blocking)  
    response \= arduino\_serial.readline().decode('utf-8').strip()  
      
    if not response:  
        raise TimeoutError("Arduino did not respond within timeout window.")  
          
    return response

## **6\. Summary**

By abstracting the raw serial communication behind the MCP interface, we effectively turn the Arduino into a **"Hardware Function Call"**. The LLM remains unaware of baud rates or voltage levels; it simply invokes toggle\_light() and the bridge handles the physical translation.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAACgElEQVR4Xu2WS2hTURCG0xet4gOEgOZ5bxJN0UqFWNGliAgq6tJNsRV8QEG6aFDqY+FO3bhxY2lFxEVFEFSsWkRXulAoXdmKlaKCiAjagi71GzwnPR1vIAlEC80PP2fOP5OZyXkloVANNSwAJJPJbfBlCXwOn/q+365zVBUUHYBjcE86nY57nrcaPmb+i/FAJpMJG/2QaLFYLKNzVBONFJ2k6CpXRPsEZ8Wv9I8MDa5WVSQSiZ2szAVXY94qKwVHXB3Uo71SWnVBM13xeDyttOPSIM2fdPVsNrsc/bSr/RfQ4LA0yGXYon0LAjT3Gc6E/uVZKxVs6wZz/h5on4M6w1JQUiy75lH7EXVfaN88ENBjGsxrnwW+m3C31oNQZmweDml9Hgi4LQ1ycTq0z0CepVm4RjsCUE6s1L4vb63WXdQR9CVZ5PyhH2YbrjH+INHlaDQasz65UPJcySowthSL5b1dgt6JdopF2My4yaRoIPZ7KpVK2Jx/QYrI6sFR7RPkcrkmkp/Df1WasDr2dbQ+zEb8R6WhIrHif8h8VyQSWYpvAvaIQ5rFfmdzFoAjgmMKTsMZ+NPwA3wLfTee+QgFDto5BXegjTnzvczHg2Kxj6E9c2Lfs6IbTWwfHLS+SiFnSr5E4Ux5f36zC4839iV4IygWe1RW1XzOY/41ZG449j3ZehtbEcyZeSM2Cc/INjG/BbtF4/ysQ39NofVBsYx30PaLRswJ7Luywug+/IaWYuyfq1gm2I4oCSbhRRrYJxpjG0WGSX7ENLu1WCz2dniFmF7G83IeGc8a3xOx+dzauYoVgOQt/P1q1jqrt1JrQbFyecLh8DLrd33ErnDnNSwK/AbG77R52zEQ7gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAACUElEQVR4Xu2VTUhUURTH36hoQl8Yk9J8vHkzAxPTQiiLkmgTgRVFq5Z9uIhUCCpC/EBCc6EQ7XIhrloF7ZIKXYQudJHQIloEBoFBVLTIoFZhv39zH12ui2ZifIvwD3/uuf9z3rnnvnvefZ63iU2UAd/3j8AXZXABPg+CoNXNURWQfBK+hKdzuVwqk8m0wBnma4zn8vl83OgXpSWTybyboxqoI/kbkjfZItoH+E1+R3/PUGtrVUE6nT7BTsdsjfle7Rw+tXVQg7bkaNUBi15KpVI5R7uqQiiy19YLhcI29AFb21BQyEMVQlMecn2RgiI+wlVvI3qhXHAc+0x/PHF9kYICekwht1xfpKCARyqEBj7o+qJEjCI+/60/dLkRMwg7aeyOeDy+1bhiug7Qh+FNCfoise+iF+EZ7FHYbaVbD30l5lhmXV8IYprxT1PMbl31ijeXoTYxq+IU55du64BCb1PAeexPxB8jthF7Zd3tTMV7cLyF7+Aq/GG4ApeVzI4n6RzJr8vG1w5fG/2K8uC7jD2E3W9i9sNhtHua87toYP6VuJY/WSsESbaT5Gc2m01rjj0A7xt7Go67vwqBRefZ8FkTdwouujEVoVgs1pPkC2ad2dmSXrt82CPY18JYFm7TruEWfN8Zd0onZgK7j/FCIpHYFcZXDHME436p6dbUK9KxAzil34P6AnZJZzys4wyfJ+YOfIB2PNT+BTUwJoNER0n4yvF7Zpe/Ywxq9VasucfR7rDnFYPFn7F4pzmWx/CkGxMJtLBfatAbFHXA9f8X+AVswJbGz+j1DgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAYAAACbU/80AAACHklEQVR4Xu2UzUuUURjFZzQwsCKMYaT5fJmBCSOE/Ig2LoQgisKVbQQzIozMTYVQEaEk6qI/oPAPEHQXRRpImzYKbhUVBIUod+OihYv6Hbpjd54Q1NekxRw43HvP87znPvfjvZFIBRWATCZzGc7tgV/gbBAEjdYjFDB9Cxfg9Vwul8pms/VwmvFP2o58Ph9zere0ZDKZtx5hcAzTJUzrfBHtK9xS3OgbNNW+FgrpdPoKKxv1NcbntFL4wddBFdq80cKByW6nUqmc0XpVAMUN+HqhUDiJ/szX/gkoYEIFcNlabexIwOTfYDFymGe9V7Dt5935v7exQ0CVFf4CEz9wBTyxsZDQ37YWj8drbaAMJE2qAC5mi42FAffqEr6LVreIkrS52/mremKv4DXejQu0zzmy4VgsdsJLi+rXJjYIH0lQDv0ZuEwhI15uOXTr3fbP2JiA3odZA+027UOnTcG7LkUL0ER3XEwvbMCkx2k/8U0Xr2rNjqHAVp8luArXYBH+cFyHKzIo5WLQxLgdLpQ0zKfRO138nrzQeui/oP/Upen8i+j1pe8OjMzvrX2tPoan6W/pgXKxd3DMPuvu/Jd87cBgZZ/hDfUxvkX/I+YXOb422iHG/aVcdrdZq0Z/DN9o+7Uzf9z2D53xd4xOaYD5VR0B7NWYWADHNYYv4X3pFHITfVYx++zvG/Y/dttf9sAkEokzNFFfc0WXaRX8F/gFKXmKcm7o8c4AAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAADOUlEQVR4Xu2WSWgUQRSGxyS4KyqOE4fM9ExmIDJuEDciHsR9BQX1IqIEiRolbohKvIhEUPEmAc1BMCoIioIxcTmoiBsJ5CAeJApiggviRcWcRL8/Xa2dIiYhc+lgfvipV/979fpVdVV3hUIDGMAAeoTjOCWwsRd8Au8nk8npdo5AgOJqYDNcmUqlYolEIh/epf+Ldk06nQ4bfbO0goKCtJ0jCMijuFcUN84von2A3+S39DaaXL8WCMTj8cWs9Am/Rn+SVh42+HWQg9ZkacEARW+JxWIpS9uuiTDJg369qKhoFHqlXws0mMgVTYRDPdv29SswiU/wayiIZ6G3YDtNNuej3vaBQeiXNNFoNDredv4LvOGlbN8ptk6eiTzvDu1L25c1SLrTTOSA7RMoag6+FlvvDsRXa5ytC/i2wmu2njVIelUTYQVn2T5BHwD8NbbeV5CrlpwVtp4ttHU+O92cD3wNrO5ZfbIpYL002oXop8223EdbJp3tN5z+Mfqb/Dn0c0U7jm8jbPPfFORDOwJLecaycDg8Ujr/uWEm/hRjV/3N1gX0ldLbgPdsn0GuJqliQ+6kH2gMDzyMXQ2f0p/Gg94Z/3592mk/egmwV8Dr3BaGJNybwhfFysfYCP06JjNBk1Mt+lHrZoHdgj1V47Bvefn+gAdFcbyBbx23yHbDVvgaJn2xMxXr9bGfUfRuxcDH2Ku1ct4VxsTvgrXqm1V9zwTmm/GljLnh5cN+iG+v8c11zEeA9jKsw7dNbxKu88b0CSQrJ8l52aaoH7DEs1mt0V2MaWLMItm08+i3a1WN7wLF7ZGtsfR/FhYWxo2vElYb+ztc622zrOG4X5gqY2u/dmxB2gWwsXN0R+EZ9FbasbTlZrs0y2fOTxssZjKHMpnMYMfdZnlm+2gBNigW+5G2rC9v92ekJ/DAfJLWx90v10XvoqlC4FE7noc7xD4nttKsps7NObQKaQn3BqGbd7Hi0cuwT8Iq+EtnRTr2cniG+B3ykXdJpwf1EXkkHOMXtLohc2BtRCKREbamN+TZ5BpqzJyQyWG24AsvxvPbN/RAguJvU3yp2VY39RbsmH4Bs310wPUfmmH7/xv8BoJ93HHnQbFAAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAYCAYAAAB5j+RNAAACXElEQVR4Xu2WO2iUQRSFN7uKilFEWFdkn+zCisZYLIpRsNEookUCYpPCB4KRgI1KEk0KQSO+sLELIqYICGLlAxQUoqRQQVtRIRBBRGzis4vfgTtmHAvjwu4PsgcOM3Pu/HfOP/+dnY3FGmggAuRyuTb4bBYch48KhcLaMEfNwILD8AXcWSwWM/l8fjm8z3iatqNUKiVN3ystnU6Xwhy1whwWfMWCS30R7T38rHigv6NJ+FrNkM1m29mRc77GeKV2CN7zdRBHex5otQNG9mUymWKgdcscxnt9vVwuL0I/6Wt1B+ZuyByFvz6MRQ6MfYBTsXrV1mzBp1xt9XY3jEUOTPWYueNhLHJg6qbMcUjWhbGo0YSxj3+rN4y3cGj6mHcxmUw2S1OL1q8TzkFqJXYUtrln+B1dwLgLXmDOLqfzzHy0A2hn4Van/wGdTvukD8KYA7EBEl63fhf9/TJG4qFKpTKX9pp2H/0Q7Yjm6Yah/xqDa+jPo39Hur3QU8YbeW4T7Rt/Le3CCsS3cAJOwe/GSU2GBW9uC+MfWlg7BC+nUqmF7srTHOJPcjPX4BLTRuFte24I7pZuL3Jafc1lvMqt9c8g0TGSjJmReBiXUeZ8Y3cW+zraF9jpSsCQQPtKrg2eVj14sy0kvOXG7M4yHRzpLHJKtUT8pWIqEbTt6qM9Vh16eVRzqu9P2mGTE+h73JxqoLt1mCRHaE/AQV1pSmqf6Dx8aPFfP0X0d8ArmD1Mewaj26SrXhlfhQfhJfTNM0tVCaul3/6t2KdsgvHwsxri4T8fQQcEU6lQb+C/xU9CHpujAqrLagAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAl0lEQVR4XmNgGAWjgHpAQUGhEF2MbAA0bKGMjIwqujhZQE5OzlpeXn4bujjZAGhYNtDQNHRxBqCThWRlZaVIxUADlwLxWhAbbhgwDDqBgstJxUCXnQTS/4B0PZLbSAdA36gADdoLCj90OZIA0CccQIOuSEtLy6DLkQyABqUAcTG6OFkAaNB+IMWCLk4WABomiS42CgYBAABUyybk/x0YCgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAXCAYAAABTYvy6AAADaUlEQVR4Xu2XSWiTQRTH09alrogYW9MkX5pGo1UUzcVIXRCXongRtQURl4qIIOpFKVpRSoUqKIigopeWqkh7cMHlULG49VAUUUGLSL0UrSdRrB7E+nv5ZtLJkEovKaL5w2Pm/ec/8+bNzDeZeDxZZJHF/4LcYDC4xnGculAotI9yii0YAuQSuwKrZy4HmMNcW0DbEmxrcXHxHMp8ytnoqqgvNnXSH668pKRkss/nmyR14UyNp7S0dASdW7BbDLQIwX7q3Vg8RZhBSBLEu4fVEn8T/g3qv6gfs3RH4fsse+SkblROGs0PbLuhSQy2E7KHlRmtOfw67C3VYYY0Y2AOR4h3gWqe5vCfq0mvsnSvsA7sErYrFosN1+0a8L3YQ+w+feqdNKdHRM9khU2OlV4mQSnLTD5TIFarxGMelZoj9mGVeJPJodmi/YFAny6bSwGDTFABG0webp4KWmPymQJxtmHt4XB4muZIcoOaQ6Ohqxlk4u9sLgV+v3+qSvy8yRN0pvCUZ01+KEH8EyrxtQZ3SB3dZsc9yre5wAJmP6V7j24vZSvlC/LYYQvi6RKEm6EW5KrJDxUikYiX+J+xl2zOKM0zn2q4p3JTi8+814kOvry/d2L+37EqqdN/Iu0P8G969J0FMV8lfs7qmEgcu2zyJgoKCsYEAgHfYEw+Kbv/nyBxsTdFRUV+k2ecQnssdF2Oe7RzNYdmuiERf7PkE9KfCasRUQnKjZoEk52l+NMmb4K2jdiVQdpBu/9AEC3WIbtut4Ecm0D72HE3L2zQKTralovG0feF1+sdi/PTPtJwC9RgqT/6GYbamdZoNDpOfDnm+vuUk0NbD/4pq48c4z7ZLPEd96f4k7kQ+EtV4i3JjjhtWHuS8CQGqxShvIxMPpMg5kpiNrPTIzWnNuCM1JlLVE3+Yn+vhKYT69W/5457wr6ax13n45gnD7IC4pusqObwG7G2pCjDcNyfzy+O+wqTF5xshtza3ezcbq2jftf87vFjkhA5VGsOfzV+vfZBDv51+A9y0Rl8QlyLvWagPZRN2BNWfnyKKINw1HeaztjpFVrH5iyEa8dOYsexj1J6Ul+YeSTaQC7XHPe+uIN1yiIZmn4gLgy6j4Z4umfgXwR5i8eZ63rmHLIbNdSfk0p0ZZT5dnsWWWSRxT+H31AwDT0STqNtAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,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>