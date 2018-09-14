from consol/ubuntu-xfce-vnc
MAINTAINER authors=whd (zhangs@live.jp)

# because base image use a non-root user we should swith to root here
USER root

# it's a bit conufusing to remove chromium here
# but it's necessary because chromium will not run under web driver but base image pre-installed it.
RUN apt-get update && apt-get install -y git unzip \
&& apt-get remove -y chromium-browser --purge

# download miniconda(python3.6) and install
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh \
&& bash ~/miniconda.sh -b -p $HOME/miniconda

# install packages we need and chrome
ENV PATH="$HOME/miniconda/bin:$PATH"
RUN echo $HOME && echo $PATH \
&& pip install opencv-python keras tensorflow hickle selenium pillow \
&& wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -   && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list   && apt-get update -qqy   && apt-get -qqy install     ${CHROME_VERSION:-google-chrome-stable}   && rm /etc/apt/sources.list.d/google-chrome.list   && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# download source and chrome web driver . use patch here because chrome can be launched only
# with -no-sandbox option by root user. Also, the function imshow of openCV have some problems
# at docker environment.
RUN echo $HOME && echo $PATH \
&& mkdir $HOME/dino \
&& cd $HOME/dino \
&& git clone https://github.com/wanghaidong1972/dino_runner.git \
&& cd dino_runner \
&& patch dinoenv.py diffs \
&& wget https://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip \
&& unzip chromedriver_linux64.zip