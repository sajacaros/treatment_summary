# 동억님의 공간
## 설치
* windows
  ```
  choco install ffmpeg-full
  ```
* linux
  - nvidia
    ```
    $ sudo yum install gcc -y
    $ sudo yum install perl -y
    $ sudo yum install kernel-headers -y
    $ sudo yum install vulkan-loader -y
    $ sudo yum install pkg-config xorg-dev -y
    $ sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
    $ sudo yum install kernel-modules-extra
    # CUDA Toolkit 12.0
    $ wget https://kr.download.nvidia.com/tesla/525.147.05/NVIDIA-Linux-x86_64-525.147.05.run
    $ sudo sh NVIDIA-Linux-x86_64-525.147.05.run
    ```
    * 주의점)
      - amazon linux 2에서 지원하는 nvidia 드라이버의 cuda 버전은 12.0과 12.2를 지원합니다.
      - torch가 지원하는 cuda 버전은 12.1을 지원합니다.
      - 윈도우즈에서 cuda 12.2설치후 torch를 사용했을때 문제가 발생했었습니다.
      - 그래서 nvidia 드라이버는 cuda 12.0을 지원하는 버전으로 설치했습니다.(높은 버전의 드라이버에서 100% 문제가 발생하는 것은 아닙니다.)
      - wget으로 스크립트를 받아오는 부분입니다.
      - 참고 드라이버와 cuda 호환성(https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)
    * cuda는 별도로 설치를 하지 않아도 됨(리눅스에서는 torch를 설치할때 자동 설치됨)
  - ffmpeg
    ``` 
    $ cd /usr/local/bin
    $ mkdir ffmpeg
    $ cd ffmpeg/
    $ wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    $ tar -xf ffmpeg-release-amd64-static.tar.xz
    $ mv ffmpeg-6.1-amd64-static/* ./
    $ ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
    $ ln -s /usr/local/bin/ffmpeg/ffprobe /usr/bin/ffprobe
    $ rmdir ffmpeg-6.1-amd64-static/
    $ rm ffmpeg-release-amd64-static.tar.xz
    ```
    * 주의점)
      - amazon linux 2의 패키지 저장소에는 ffmpeg이 없습니다.
      - 직접 소스를 빌드하거나 빌드된 파일을 다운 받아야 합니다.
      - 빌드된 파일을 다운 받아 진행했습니다.
      - 명령어가 길지만 요약하면 다운 받은 후 실행파일이 실행될 수 있도록 path 등록된 곳에 링크(소프트)를 걸었습니다.
* pipenv 환경 들어가기
  ``` 
  $ cd ~/workspace/service
  $ pipenv shell
  $ cd treatment-summary
  ```
* package
  ```
  $ pip install openai
  $ pip install python-dotenv
  $ pip install pydub
  $ pip install streamlit
  $ pip install -U git+https://github.com/jianfch/stable-ts.git
  $ pip install ffmpeg-python
  $ pip install soundfile
  $ pip install nlptutti
  $ pip install fastapi
  $ pip install "uvicorn[standard]"
  $ pip install streamlit-authenticator
  $ pip install langchain
  $ pip install langchain_google_genai
  
  # option
  $ pip install memory_profiler
  $ pip install moviepy
  $ pip install chardet
  $ pip install noisereduce
  $ pip install spleeter
  $ pip install --force-reinstall charset-normalizer==3.1.0
  $ pip install malaya-speech
  $ pip install denoiser
  $ pip install --upgrade typer 
  ```
* torch
  - [install torch locally](https://pytorch.org/get-started/locally/)
  - `libcudnn_cnn_infer.so.8`을 못찾는다는 오류 메시지가 뜬다면 아래 path를 적절히 수정하여 `~/.bash_profile`에 추가
  ```
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/.local/share/virtualenvs/service-9_XJoB3L/lib/python3.9/site-packages/nvidia/cudnn/lib
  ```
* gemini key 받기
  - https://makersuite.google.com/app/apikey 방문
  - `Create API key in new project` 클릭
* .env 설정
  - `DATA_PATH`는 audio가 들어있는 절대경로로 설정
  ```
  OPENAI_API_KEY={openai_api_key}
  GOOGLE_API_KEY={google_api_key}
  DATA_PATH={root_path}  ex> 'C:\Users\dukim\workspace\treatment-summary\data\'
  HUGGING_FACE_KEY={huggin_face_key}
  ```
* 실행
  
  - 설치
  ``` 
  $ sudo yum install tmux
  ```
  - 세션 생성
  ```
  $ tmux new -s charting-api
  $ tmux new -s charting-ui
  ```
  - 세션 빠져 나오기
  ``` 
  cntl+b -> d 
  ```
  - 세션 조회
  ``` 
  $ tmux ls
  ```
  - 세션 다시 들어가기
  ``` 
  $ tmux a -t charting-api
  $ tmux a -t charting-ui
  ```
  - charting api 실행
  ``` 
  $ tmux a -t charting-api
  $ python ./charting_api.py
  ```
  - charting ui 실행
  ``` 
  $ tmux a -t charting-ui
  $ streamlit run ./charting_ui.py --server.port 8080 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
  ```
* NVIDIA CONTAINER TOOLKIT
  - ![](../images/nvidia_docker.png)
  ```
  $ curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  $ sudo yum install -y nvidia-container-toolkit
  $ sudo touch /etc/docker/daemon.json
  $ sudo nvidia-ctk runtime configure --runtime=docker
  $ sudo systemctl restart docker.service
  $ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
  ```
* docker build
  - build(in Dockerfile directory)
  ```
  $ docker image build -t charting-app -f docker/Dockerfile .
  ```
  - execute
  ```
  $ docker run -p 8080:8080 --memory=7g --memory-swap=14g --gpus all --name charting-app -v C:\Users\dukim\workspace\data:/data charting-app
  $ docker run -d -p 8080:8080 --memory=6g --memory-swap=10g --gpus all --name charting-app -v /home/ec2-user/workspace/data:/data sajacaros/treatment-summary:latest
  ```
  - tagging
  ```
  $ docker tag charting-app sajacaros/treatment-summary:v1.3
  $ docker tag charting-app sajacaros/treatment-summary:latest
  ```
  - docker hub push/pull
  ```
  $ docker push sajacaros/treatment-summary:v1.3
  $ docker push sajacaros/treatment-summary:latest
  $ docker pull sajacaros/treatment-summary:latest
  ```
* 배포 설정
- 서버에서 키 생성
``` 
$ ssh-keygen
```
- 키 확인
``` 
$ ls ~/.ssh/
```
- 깃허브에 키 등록
  - https://github.com/bootcamp6th-nlp/treatment-summary/settings
  - -> Deploy keys( https://github.com/bootcamp6th-nlp/treatment-summary/settings/keys )
  - -> Add deploy key
  - -> 복사한 공개키 붙여 넣기
  - 공개키 복사하기
  ``` 
  $ cat ~/.ssh/id_rsa.pub 
  ```
- ssh 설정
``` 
$ vi ~/.ssh/config
Host github.com-treatment-summary
        Hostname github.com
        IdentityFile=/home/ec2-user/.ssh/authorized_keys
Host github.com
        Hostname ssh.github.com
        Port 443
```
- 깃 clone
``` 
$ git clone git@github.com:bootcamp6th-nlp/treatment-summary.git 
```
* 인증
- 비번 생성
``` 
import streamlit_authenticator as stauth
HASHED_PASSWORD = stauth.Hasher(['PASSWORD']).generate()
```
- auth yaml 생성
``` 
credentials:
  usernames:
    medistream:
      name: Admin
      password: HASHED_PASSWORD
# nlp2th
cookie:
  expiry_days: 30
  key: whisperandllm # Must be string
  name: treatment-summary
```