# Usa Ubuntu 22.04 como base
FROM ubuntu:22.04

# Evita preguntas interactivas durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Actualiza el sistema e instala dependencias
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    unzip \
    zip \
    openjdk-11-jdk \
    autoconf \
    automake \
    libtool \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libltdl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    libexpat1-dev \
    libc6-dev \
    libbz2-dev \
    libssl-dev \
    ccache \
    lld \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala Buildozer desde PyPI
RUN pip3 install --upgrade pip Cython
RUN pip3 install --upgrade buildozer

# Instala dependencias adicionales de Android
RUN apt-get update && apt-get install -y \
    libltdl-dev \
    libffi-dev \
    libssl-dev \
    autoconf \
    automake \
    libtool \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-descarga las dependencias de Android de buildozer
RUN mkdir -p /root/.buildozer/android/platform
WORKDIR /tmp
RUN pip3 install -v pyjnius
RUN apt-get update && apt-get install -y cmake libstdc++6 && apt-get clean
RUN pip3 install cython

# Crea un directorio de trabajo
WORKDIR /github/workspace

# Entrypoint para permitir ejecutar comandos arbitrarios
ENTRYPOINT []
