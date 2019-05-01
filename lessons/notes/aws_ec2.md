# Remote EC2 Controls

**1. Start EC2 instance**

```
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**2. Find IPv4 Public IP**

> Note: The instance IP is not part of the start-instances response. It can be found in the EC2 console.

**3. SSH into the instance (using the key-pair pem). Bind port 8000 to the remote Jupyter port 8888 to access the remote notebook locally**


```
ssh -i crodriguez1a-udacity-keypair.pem -L 8000:localhost:8888 ubuntu@3.83.173.81

```

**4. Access the notebook via port 8000**

```
http://localhost:8000/tree/
```

**5. Stop the instance**

```
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
```
