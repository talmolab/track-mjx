# Directly from a cuda built image.
FROM nvidia/cuda:12.6.1-base-ubuntu24.04  

LABEL maintainer="Scott Yang <scyang@salk.edu>"

USER root

RUN apt clean
RUN apt update


RUN apt update --yes && apt install --yes build-essential
RUN apt-get update && apt-get install -y openssh-server

# use tini instead of init
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]


# SSH
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# install conda
RUN apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "Miniforge3-Linux-x86_64.sh" && \
    chmod +x "Miniforge3-Linux-x86_64.sh" && \
    bash "Miniforge3-Linux-x86_64.sh" -b -p "/root/miniforge3" && \
    rm "Miniforge3-Linux-x86_64.sh" && \
    /root/miniforge3/bin/conda init bash && \
    . ~/.bashrc

# add conda to path so we can create env
ENV PATH "/root/miniforge3/bin:${PATH}"

# install conda env
RUN mkdir track-mjx
COPY . ./track-mjx
RUN conda env create -f track-mjx/environment.yml
