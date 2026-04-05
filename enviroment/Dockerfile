FROM registry.codeocean.com/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- Script to install project fonts at runtime (Code Ocean compatible) ---
RUN printf '#!/usr/bin/env bash\n\
set -e\n\
SRC=/data/fonts/Arial\n\
DST=/usr/local/share/fonts/truetype/arial\n\
mkdir -p "$DST"\n\
if [ -d "$SRC" ]; then\n\
  cp -f "$SRC"/*.[Tt][Tt][Ff] "$DST"/ || true\n\
fi\n\
fc-cache -f -v\n' > /usr/local/bin/install_project_fonts && chmod +x /usr/local/bin/install_project_fonts

ENV VENV=/opt/venv
RUN python -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

RUN pip install -U pip setuptools wheel

RUN pip install --only-binary=:all: \
    "numpy==1.26.*" \
    "pandas==2.2.*" \
    "xarray==2024.3.*" \
    "scipy==1.11.*" \
    "matplotlib==3.8.*" \
    "seaborn" \
    "pymannkendall" \
    "statsmodels" \
    "netCDF4==1.6.*" \
    "h5netcdf==1.3.*" \
    "openpyxl==3.1.*" \
    "regionmask==0.12.*"\
    "brokenaxes" \
    "ipython"\
    \
    "pyproj==3.6.*" \
    "shapely==2.0.*" \
    "rasterio==1.3.*" \
    "geopandas==0.14.*" \
    "cartopy==0.22.*" \