#Ubuntu with a desktop and VNC server.  Will allow us to use gui monitor tools that these libaries have such as tensorboard
FROM consol/ubuntu-xfce-vnc

#switches default user to root
USER 0

#use bash for the shell
SHELL [ "/bin/bash","-c" ]

#sets the environment so that we can use conda after install
ENV PATH="~/anaconda3/condabin:${PATH}"

#Used for GPU setup
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

#Needed to build some packages
RUN apt update \
&& apt install gcc -y \
&& apt install build-essential -y \
&& apt install unzip -y \
&& apt install nomacs -y \
&& apt install git -y \
&& apt install nano -y \
&& apt install gedit -y \
&& apt install imagemagick -y


#gets the anaconda package, installs it, and adds it to the path
#gets the anaconda package
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
#sets the correct permissions to allow install
&& chmod 777 Anaconda3-2019.03-Linux-x86_64.sh \
#silently installs the package
&& ./Anaconda3-2019.03-Linux-x86_64.sh -b  \
#sets the path, may not be needed with the ENV PATH being used above
&& echo "export PATH=\"/headless/anaconda3/condabin:$PATH\"">>.bashrc \
&& source ~/.bashrc

RUN conda init \
&& conda create -n diabetic python \
&& conda install -n diabetic matplotlib=3.0.3 \
&& conda install -n diabetic opencv \
&& conda install -n diabetic pandas \
&& conda install -n diabetic scikit-learn \
&& conda install -n diabetic tensorflow-gpu=1.14 \
&& conda install -n diabetic Pillow  \
&& conda install -n diabetic pydot \ 
&& conda install -n diabetic seaborn \
&& conda install -n diabetic h5py


RUN conda init bash \
&& source ~/.bashrc \
&& conda activate diabetic \ 
&& pip install -U efficientnet

RUN conda init \
&& conda create -n diabetic2 python \
&& conda install -n diabetic2 matplotlib=3.0.3 \
&& conda install -n diabetic2 opencv \
&& conda install -n diabetic2 pandas \
&& conda install -n diabetic2 scikit-learn \
&& conda install -n diabetic2 tensorflow-gpu=2.0.0 \
&& conda install -n diabetic2 Pillow  \
&& conda install -n diabetic2 pydot \ 
&& conda install -n diabetic2 seaborn \
&& conda install -n diabetic2 h5py

RUN conda init bash \
&& source ~/.bashrc \
&& conda activate diabetic2 \ 
&& pip install -U efficientnet \
&& pip install tensorflow-addons==0.6.0


RUN conda init \
&& conda create -n no_gpu python \
&& conda install -n no_gpu matplotlib=3.0.3 \
&& conda install -n no_gpu opencv \
&& conda install -n no_gpu pandas \
&& conda install -n no_gpu scikit-learn \
&& conda install -n no_gpu tensorflow=2.0.0 \
&& conda install -n no_gpu Pillow  \
&& conda install -n no_gpu pydot \ 
&& conda install -n no_gpu seaborn

RUN conda init bash \
&& source ~/.bashrc \
&& conda activate no_gpu \ 
&& pip install -U efficientnet \
&& pip install tensorflow-addons==0.6.0