# XR Cognitive Assistant

## Getting Started

To install dependencies for the server, run:
```
pip3 install -r requirements.txt
```

## Serving

In a new Terminal, generate certs:
```
mkdir ssl
openssl req -x509 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem -days 365
openssl rsa -in ssl/key.pem -out ssl/newkey.pem && mv ssl/newkey.pem ssl/key.pem
```

Then, run:
```
python3 server.py --host <YOUR IP ADDRESS> --cert-file ssl/cert.pem --key-file ssl/key.pem
```

On the ML2, set the `Server Address` of the `WebRTCConnection` component to `https://<YOUR IP ADDRESS>:8000`
