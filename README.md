# subtitle-sync
subtitle-sync repository contains simple app with for subtitle file generation from text and audio.


### Project Setup

#### Clone the Repository


```bash
git clone https://github.com/rizwanahmad8311/subtitle-sync.git
cd subtitle-sync
```

The required version of python for this project is 3.10.Make sure you have the correct version.
### Set up Virtual Environment

#### Install Virtualenv

```bash
sudo apt update
sudo apt install python3-venv
```

##### Create Virtual Environment

```bash
python3 -m venv venv
```

##### Activate Virtual Environment

```bash
source venv/bin/activate
```


#### Install Requirements

```bash
pip install -r requirements.txt
```

#### Running the Server

```bash
python app.py
```
### Subtitle Sync APP
You can now access the app:

* [Subtitle Sync APP](http://127.0.0.1:7860/)

## Dockerized Server

### Usage

#### Build the Docker Image


Open cmd/shell and change location where `Dockerfile` is located and run the following command. This may take a while (6-10 minutes) depending upon internet speed.

```shell
docker build -t subtitle-sync .
```

* `-t subtitle-sync` names your image `subtitle-sync`
* `.` means Dockerfile is in the current directory

#### Run the Docker Container

```shell
docker run -p 7860:7860 subtitle-sync
```

#### Run in Detached Mode

```shell
docker run -d -p 7860:7860 --name subtitle-container subtitle-sync
```

Run the following command to check the running containers

```shell
docker ps
```

#### Environment Variables

* `-d` - This command starts the container in the background, allowing you to use your terminal freely.

### Subtitle Sync APP
You can now access the app:

* [Subtitle Sync APP](http://127.0.0.1:7860/)
