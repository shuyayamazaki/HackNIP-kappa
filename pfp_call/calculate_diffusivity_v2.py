import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def calculate_diffusivities(list_to_label):
    """
    Calculate diffusivities for multiple CIF contents with temperatures in parallel.

    Args:
        list_to_label (list of tuple): List where each element is a tuple containing:
            - CIF content as a string.
            - Temperature (float or int) at which AIMD will be performed.

    Returns:
        list: Diffusion coefficients (D) in cm^2/sec for each CIF content-temperature pair.
    """
    print("[INFO] Installing dependencies on remote server.")
    # Install dependencies once
    install_dependencies()

    temp_out_path = "temp.out"
    if os.path.exists(temp_out_path):
        os.remove(temp_out_path)
        print(f"[INFO] Removed existing temp.out file")

    with open(temp_out_path, 'w') as temp_out_file:
        diffusivities = []

        def process_item(index, cif, temperature):
            temp_cif_path = f"temp_{index}.cif"

            print(f"[INFO] Saving temp_{index}.cif file.")
            # Save CIF string to temporary file
            with open(temp_cif_path, 'w') as f:
                f.write(cif)

            # Run remote script and get the output
            diffusivity = run_remote_script(temp_cif_path, temperature, index)

            # Log diffusivity to temp.out file
            temp_out_file.write(f"Index: {index}, Diffusivity: {diffusivity:.12f} cm^2/sec\n")
            temp_out_file.flush()

            return diffusivity

        # Run calculations in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda item: process_item(item[0], *item[1]), enumerate(list_to_label, start=1)))
            diffusivities.extend(results)

    return diffusivities

def install_dependencies():
    """
    SSH to a remote host and install dependencies.
    """
    try:
        requirements_file = "/home/jovyan/SKim/Diffusivity/StrToIcon/requirements.txt"
        install_command = f"/home/jovyan/.py38/bin/pip install -r {requirements_file}"
        ssh_command = f"ssh matlantis \"{install_command}\""
        process = subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        if process.returncode != 0:
            raise RuntimeError(f"Failed to install dependencies: {process.stderr}")

    except Exception as e:
        raise RuntimeError(f"Error occurred while installing dependencies: {e}")

def run_remote_script(temp_cif_path, temperature, index):
    """
    SSH to a remote host, upload temp.cif, run a Python script, and save the diffusion coefficient in temp.out.

    Args:
        temp_cif_path (str): Local path to the temp.cif file to be uploaded.
        temperature (float or int): Temperature at which AIMD will be performed.
        index (int): Unique index for the CIF file to distinguish remote paths.

    Returns:
        float: Diffusion coefficient (D) in cm^2/sec.
    """
    try:
        # Remote paths
        remote_cif_path = f"/home/jovyan/SKim/Diffusivity/StrToIcon/input/temp_{index}.cif"
        diffusivities_script = "/home/jovyan/SKim/Diffusivity/StrToIcon/diffusivity_calc.py"

        print(f"[INFO] Uploading temp_{index}.cif to remote server.")
        # 1. Upload temp.cif to the remote server
        upload_command = f"scp {temp_cif_path} matlantis:{remote_cif_path}"
        upload_process = subprocess.run(upload_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        if upload_process.returncode != 0:
            raise RuntimeError(f"Failed to upload temp.cif: {upload_process.stderr}")

        print(f"[INFO] Running remote script for temp_{index}.cif.")
        # 2. SSH to the server to run the script with temperature parameter
        run_command = f"/home/jovyan/.py38/bin/python {diffusivities_script} --temperature {temperature} --index {index}"
        ssh_command = f"ssh matlantis \"{run_command}\""

        process = subprocess.Popen(
            ssh_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Remote command failed: {stderr}")

        print("[INFO] Parsing diffusion coefficient from script output.")
        # Extract the last floating-point value from the output
        last_line = stdout.strip().splitlines()[-1]
        diffusivity = float(last_line)
        print(f"[INFO] Extracted diffusion coefficient: {diffusivity:.12f} cm^2/sec")
        return diffusivity

    except Exception as e:
        raise RuntimeError(f"Error occurred while running remote script: {e}")

# Example usage
if __name__ == "__main__":
    list_to_label = [
        ("""CIF_Content""", 1000), # Temperature in K
        ("""CIF_Content""", 1500)
    ]

    try:
        print("[INFO] Starting diffusivity calculations.")
        results = calculate_diffusivities(list_to_label)
    except Exception as error:
        print(f"[ERROR] An error occurred: {error}")
