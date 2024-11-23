import platform
import subprocess

import numpy as np
import psutil


def get_cpu_info():
    cpu_info = {}

    cpu_info["Model"] = platform.processor() or "Unknown"

    freq = psutil.cpu_freq()
    if freq:
        cpu_info["Base Frequency"] = f"{freq.min:.1f} GHz"
        cpu_info["Max Frequency"] = f"{freq.max:.1f} GHz"

    cpu_info["Cores"] = psutil.cpu_count(logical=False)
    cpu_info["Threads"] = psutil.cpu_count(logical=True)

    try:
        lscpu_output = subprocess.check_output("lscpu", shell=True).decode()
        for line in lscpu_output.split("\n"):
            if "L1d cache" in line:
                cpu_info["L1 Cache"] = line.split(":")[1].strip()
            elif "L2 cache" in line:
                cpu_info["L2 Cache"] = line.split(":")[1].strip()
            elif "L3 cache" in line:
                cpu_info["L3 Cache"] = line.split(":")[1].strip()
    except Exception:
        cpu_info["L1 Cache"] = "Unknown"
        cpu_info["L2 Cache"] = "Unknown"
        cpu_info["L3 Cache"] = "Unknown"

    return cpu_info


def get_memory_info():
    total_ram = 2 * 2  # Two modules of 2GB each
    ram_type = "DDR3"
    ram_speed = "1333 MT/s"  # got from dmidecode tool
    return f"{total_ram}GB {ram_type} {ram_speed}"


def get_compiler_info():
    # Get the default compiler and version
    compiler = (
        subprocess.check_output("clang --version", shell=True).decode().split("\n")[0]
    )
    compiler_flags = "-O2 -mno-avx512f -march=native"  # Mocked as these are custom
    return compiler, compiler_flags


def get_os_info():
    return f"{platform.system()} {platform.release()} {platform.version()}"


def main():
    cpu_info = get_cpu_info()
    ram_info = get_memory_info()
    numpy_version = np.__version__
    compiler, compiler_flags = get_compiler_info()
    os_info = get_os_info()

    # Print in the desired format
    print(
        f"CPU: {cpu_info['Model']} {cpu_info['Cores']} Cores, {cpu_info['Threads']} Threads"
    )
    print(f"Freq: {cpu_info.get('Base Frequency', 'Unknown')}")
    print(f"Turbo Freq: {cpu_info.get('Max Frequency', 'Unknown')}")
    print(f"L1 Cache: {cpu_info.get('L1 Cache', 'Unknown')} (per core)")
    print(f"L2 Cache: {cpu_info.get('L2 Cache', 'Unknown')} (per core)")
    print(f"L3 Cache: {cpu_info.get('L3 Cache', 'Unknown')}")
    print(f"RAM: {ram_info}")
    print(f"Numpy {numpy_version}")
    print(f"Compiler: {compiler}")
    print(f"Compiler flags: {compiler_flags}")
    print(f"OS: {os_info}")


if __name__ == "__main__":
    main()
