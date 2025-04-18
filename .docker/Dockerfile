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

# Instala herramientas de compilación Android
RUN mkdir -p /opt/android
WORKDIR /opt/android

# Crea un directorio para tu aplicación
RUN mkdir -p /app
WORKDIR /app

# Instala dependencias adicionales de Buildozer
RUN buildozer init
RUN apt-get update && buildozer -v android debug || true
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Establece el entorno de compilación para Android
ENV PATH=$PATH:/opt/android/android-sdk/tools:/opt/android/android-sdk/platform-tools

# Establece el punto de entrada del contenedor
ENTRYPOINT ["buildozer"]
CMD ["android", "debug"]
