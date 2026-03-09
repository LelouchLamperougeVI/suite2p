# HaoRan's bastardized suite2p

My modified suite2p including analysis pipeline.

## Installation

I'm using ```python 3.11```. This is how you install it under Linux; Windows users can go suck a nut :heart:

```console
foo@bar:~$ python --version
Python 3.11.5
foo@bar:~$ git clone https://github.com/LelouchLamperougeVI/suite2p.git
foo@bar:~$ cd suite2p
foo@bar:~$ python -m venv .venv
foo@bar:~$ source .venv/bin/activate
foo@bar:~$ pip install -e .
```

Subsequently, you just need to ```source .venv/bin/activate``` every time you need to run suite2p.

## Troubleshoot

### libKSG

If you get an error related to ```libKSG``` while installing, then probably you're missing the GNU Scientific Library.
Install it using your distro's package manager.
See [libKSG](https://github.com/LelouchLamperougeVI/libKSG).

### torch
