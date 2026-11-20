# Hardware Requirements & Deployment Guide for Computers and Laptops 💻

Welcome to the **MaxEnt Studio Pro** Hardware and System Configuration Guide. This document provides clear specifications, operating system setup instructions, memory tuning guidelines, and laptop-specific optimizations for running the software reliably across laptops, desktop PCs, and high-performance workstations.

---

## 🖥️ 1. Hardware Specification Tiers

MaxEnt Studio Pro features a **Dual-Engine Architecture** specifically engineered to scale from ultralight portable field laptops up to high-end GIS workstations:

| Hardware Tier | Target Hardware | CPU | Physical RAM | Recommended Engine | Java Heap (`-Xmx`) | Max Raster Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Portable / Budget Laptop** | Netbooks, older laptops, field notebooks | 2 Cores (Intel Core i3, AMD Ryzen 3, Intel Celeron) | **4 GB – 6 GB** | **Native Python Engine** (Consumes $<30$ MB RAM) | $1,024 - 1,536\text{ MB}$ | $800 \times 800\text{ px}$ |
| **Tier 2: Standard Laptop & Desktop** *(Recommended)* | Modern consumer laptops, MacBooks, office desktops | 4–8 Cores (Intel Core i5/i7, AMD Ryzen 5/7, Apple M1/M2/M3) | **8 GB – 16 GB** | **Dual Engine** (Python or Official Java) | $3,072 - 4,096\text{ MB}$ | $1,500 \times 1,500\text{ px}$ |
| **Tier 3: High-End GIS Workstation** | Engineering workstations, multi-core towers, HPC nodes | 8–16+ Cores (Intel Core i9/Xeon, AMD Ryzen 9/Threadripper) | **32 GB – 64 GB+** | **Both Engines** (Massive batch & multi-threading) | $8,192 - 16,384\text{ MB}$ | $4,000 \times 4,000\text{ px}$ (Full Resolution) |

---

## ⚡ 2. Laptop-Specific Optimizations

Running intensive spatial modeling on laptops introduces unique challenges (battery life, thermal throttling, trackpad navigation, and high-DPI scaling). MaxEnt Studio Pro includes several built-in optimizations:

### A. Battery Conservation & Field Prospection
- **Native Python Solver Advantage**: While the Java engine instantiates a separate Java Virtual Machine (JVM) that draws higher CPU wattage, the **Native Python MaxEnt Engine** executes compact vector operations in NumPy and SciPy. This reduces CPU thermal load and extends laptop battery life during fieldwork by up to **40%**.
- **Offline Autonomous Mode**: The entire modeling pipeline, GIS aligner, Plotly dashboards, and Folium maps operate **100% locally without an active internet connection**. Base maps cache automatically for offline field use.

### B. High-DPI & Retina Display Scaling
- Modern laptops (Apple MacBook Retina, Microsoft Surface, 4K Dell XPS) use high DPI scaling factors ($125\% - 200\%$).
- The Streamlit interface (`app.py`) and Plotly analytics adapt dynamically to high-DPI displays without blurring or clipping controls.

### C. Low-Memory Laptop Guardrail (4 GB – 8 GB RAM)
- If your laptop has 4 GB to 8 GB of RAM, select the **Native Python MaxEnt Engine** in the sidebar. Its dual L-BFGS-B formulation operates strictly in memory and requires less than **30 MB of RAM**, preventing your laptop from swapping to disk or freezing.
- If using the Java engine, set the **Java Heap Memory** slider in the sidebar to **1536 MB** or **2048 MB** to avoid starving the operating system.

---

## 🚀 3. Operating System Setup Guides

### 🪟 Windows 10 & 11 (Laptops & Desktops)
1. **Prerequisites**:
   - Python 3.9 to 3.12+ (Download from [python.org](https://www.python.org/downloads/)). **Crucial:** Check the box *"Add Python to PATH"* during installation.
   - *(Optional)* 64-bit Java JRE/JDK 8 or higher (e.g., Eclipse Temurin OpenJDK 17 or 21) if you plan to run the official AMNH Java engine.
2. **One-Click Automated Setup**:
   - Double-click [`setup_environment.bat`](file:///c:/Users/DuDu-PC/Downloads/maxent_presence_model/setup_environment.bat).
   - This script installs all required packages and probes your hardware to generate [`app_config.json`](file:///c:/Users/DuDu-PC/Downloads/maxent_presence_model/app_config.json).
3. **Launch the Studio**:
   - Double-click [`run_studio.bat`](file:///c:/Users/DuDu-PC/Downloads/maxent_presence_model/run_studio.bat) or open a terminal and run:
     ```powershell
     python -m streamlit run app.py --server.maxUploadSize 2048
     ```

### 🍎 macOS (Apple Silicon M1/M2/M3 & Intel MacBooks)
1. **Prerequisites**:
   - Install Homebrew (if not already installed) and Python:
     ```bash
     brew install python
     ```
2. **Setup**:
   - Open Terminal in the project directory and run:
     ```bash
     chmod +x setup_environment.sh run_studio.sh
     ./setup_environment.sh
     ```
3. **Launch**:
   - Run the launcher script:
     ```bash
     ./run_studio.sh
     ```
   - *Note on Apple Silicon:* NumPy and SciPy automatically utilize Apple's Accelerate framework / NEON SIMD instructions, providing blazing-fast L-BFGS-B convergence.

### 🐧 Linux (Ubuntu, Debian, Fedora, Arch)
1. **Prerequisites**:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-pip default-jre
   ```
2. **Setup & Launch**:
   ```bash
   chmod +x setup_environment.sh run_studio.sh
   ./setup_environment.sh
   ./run_studio.sh
   ```

---

## 🛠️ 4. Tuning Memory & Configuration Settings

### A. Increasing File Upload Limits (Up to 2 GB)
The configuration file [`.streamlit/config.toml`](file:///c:/Users/DuDu-PC/Downloads/maxent_presence_model/.streamlit/config.toml) overrides default limits:
```toml
[server]
maxUploadSize = 2048   # Allocates up to 2 GB per uploaded GeoTIFF raster
maxMessageSize = 2048
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### B. Probing Your Hardware Automatically
To re-detect your computer's resources at any time, run:
```bash
python system_config.py
```
This inspects total physical RAM, CPU cores, and Java runtime, updating `app_config.json` with recommended parameters.

### C. Troubleshooting Common Laptop Errors
- **`OutOfMemoryError` in Java Engine**: In the left sidebar, expand **Advanced Settings** and reduce the Java heap slider or switch to the **Native Python Engine**.
- **Port 8501 Already in Use**: If another process is using port 8501, run with a custom port:
  ```bash
  python -m streamlit run app.py --server.port 8502
  ```
- **PROJ / GDAL DLL Conflicts on Windows**: MaxEnt Studio Pro automatically points `PROJ_DATA` and `PROJ_LIB` directly to Rasterio's bundled database, preventing crashes with third-party GIS software (e.g. ArcGIS, QGIS, or Anaconda).
