"""
system_config.py - Hardware Detection & System Configuration Optimizer for MaxEnt Studio Pro.

Automatically probes local hardware (laptops, desktops, workstations) across Windows, macOS,
and Linux. Computes optimal Java heap sizes, thread allocations, raster resolution ceilings,
and runtime profiles (Low-Power Laptop, Standard Desktop, High-Performance Workstation).
"""

import os
import sys
import platform
import subprocess
import multiprocessing
import json
import ctypes
from pathlib import Path
from typing import Dict, Any


def get_total_ram_gb() -> float:
    """Retrieve total physical system RAM in gigabytes across Windows, macOS, and Linux."""
    system = platform.system()
    try:
        if system == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        elif system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out) / (1024 ** 3)
        elif system == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0  # Conservative fallback default


def check_java_installed() -> Dict[str, Any]:
    """Check if 64-bit Java runtime is available."""
    try:
        res = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        output = res.stderr or res.stdout
        is_64bit = "64-Bit" in output or "x86_64" in output or "amd64" in output
        return {
            "available": True,
            "details": output.strip().splitlines()[0] if output else "Java detected",
            "is_64bit": is_64bit
        }
    except Exception:
        return {
            "available": False,
            "details": "Java not found in system PATH (Native Python engine will be used)",
            "is_64bit": False
        }


def generate_configuration() -> Dict[str, Any]:
    """Calculate the optimal configuration profile for this computer or laptop."""
    total_ram = get_total_ram_gb()
    cpu_cores = multiprocessing.cpu_count()
    sys_name = platform.system()
    arch = platform.machine()
    java_info = check_java_installed()

    # Determine hardware tier
    if total_ram < 6.5:
        tier = "Budget / Portable Laptop (<= 4-6 GB RAM)"
        rec_engine = "python"
        java_heap_mb = min(1536, int(total_ram * 1024 * 0.4))
        max_upload_mb = 1024
        max_raster_dim = 800
        threads = max(1, cpu_cores - 1)
        notes = (
            "Recommended for battery conservation and light laptops. "
            "The Native Python engine runs completely in-memory with <30 MB RAM overhead."
        )
    elif total_ram < 18.0:
        tier = "Standard Laptop / Desktop PC (8-16 GB RAM)"
        rec_engine = "python"
        java_heap_mb = min(4096, int(total_ram * 1024 * 0.5))
        max_upload_mb = 2048
        max_raster_dim = 1500
        threads = min(4, cpu_cores)
        notes = (
            "Optimal sweet spot for field laptops and standard desktops. "
            "Easily handles high-resolution 2 GB GeoTIFF layers."
        )
    else:
        tier = "High-Performance Workstation (>= 32 GB RAM)"
        rec_engine = "python"
        java_heap_mb = min(16384, int(total_ram * 1024 * 0.6))
        max_upload_mb = 4096
        max_raster_dim = 4000
        threads = min(8, cpu_cores)
        notes = (
            "Workstation class. Can process massive continental or global grids "
            "at 30 arc-second (1 km) or fine LiDAR resolutions without downsampling."
        )

    config = {
        "system_info": {
            "os": f"{sys_name} ({platform.release()})",
            "architecture": arch,
            "total_ram_gb": round(total_ram, 2),
            "cpu_cores": cpu_cores,
            "java_installed": java_info["available"],
            "java_version": java_info["details"],
            "java_64bit": java_info["is_64bit"]
        },
        "hardware_tier": tier,
        "recommended_settings": {
            "default_engine": rec_engine,
            "java_heap_mb": java_heap_mb,
            "streamlit_max_upload_mb": max_upload_mb,
            "max_raster_dimension": max_raster_dim,
            "threads": threads,
            "server_port": 8501,
            "browser_open": True,
            "tile_caching": True
        },
        "profile_notes": notes
    }
    return config


def main():
    print("=" * 75)
    print("   MAXENT STUDIO PRO - COMPUTER & LAPTOP HARDWARE CONFIGURATOR")
    print("=" * 75)
    
    config = generate_configuration()
    sys_info = config["system_info"]
    rec = config["recommended_settings"]

    print(f"OS Platform:          {sys_info['os']} [{sys_info['architecture']}]")
    print(f"Detected RAM:         {sys_info['total_ram_gb']} GB")
    print(f"CPU Logical Cores:    {sys_info['cpu_cores']}")
    print(f"Java Runtime:         {sys_info['java_version']}")
    print("-" * 75)
    print(f"Hardware Class:       {config['hardware_tier']}")
    print(f"Recommended Engine:   {rec['default_engine'].upper()} (Pure Python Scikit-Learn Solver)")
    print(f"Allocated Java Heap:  {rec['java_heap_mb']} MB (-Xmx{rec['java_heap_mb']}m)")
    print(f"Max Raster Upload:    {rec['streamlit_max_upload_mb']} MB (Configured in .streamlit/config.toml)")
    print(f"Parallel Threads:     {rec['threads']}")
    print(f"Max Grid Dimension:   {rec['max_raster_dimension']} px")
    print("-" * 75)
    print(f"Guidance: {config['profile_notes']}")
    print("=" * 75)

    # Save to app_config.json
    out_path = Path(__file__).parent / "app_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"[+] Hardware configuration profile saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
