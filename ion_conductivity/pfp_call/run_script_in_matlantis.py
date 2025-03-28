import subprocess
import sys
import re

def run_remote_script():
    """
    SSH to a remote host, install dependencies, run a Python script, 
    and return the "D" value.
    """

    try:
        # Build the SSH command
        requirements_file = "/home/jovyan/SKim/Diffusivity/MD_Li_diffusion_in_LGPS/requirements.txt"
        diffusivities = "/home/jovyan/SKim/Diffusivity/MD_Li_diffusion_in_LGPS/diffusivity_modified.py"

        # Install dependencies using the virtual environment's pip
        install_command = f"/home/jovyan/.py38/bin/pip install -r {requirements_file}"

        # Run the Python script using the virtual environment's python
        run_command = f"/home/jovyan/.py38/bin/python {diffusivities}"

        # Combine the commands into a single SSH command
        ssh_command = f"ssh matlantis {run_command}"

        # Execute the SSH command
        process = subprocess.Popen(ssh_command, shell=True, 
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, universal_newlines=True)

#         for line in iter(process.stdout.readline, ""):
#             print(line, end="")

#         # Get the output and error
#         output, error = process.communicate()
#         # output = output.decode().strip()
#         # print(output)
#         # print(error)

#         match = re.search(r"Activation energy:\s*(\d+\.?\d*)\s*meV", output)
#         if match:
#             result = float(match.group(1))
#         else:
#             result = None

#         return result

#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
# if __name__ == "__main__":

#     result = run_remote_script()

#     print(f"value: {result}")

        # Capture the output in a variable
        output = ""
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            output += line

        # Wait for the process to finish and get the return code
        process.stdout.close()
        return_code = process.wait()

        if return_code:
            print(f"SSH command exited with code {return_code}")
            print(process.stderr.read())
            return None

        # Extract the E_act value (in meV)
        match = re.search(r"Activation energy:\s*(\d+\.?\d*)\s*meV", output)
        if match:
            result = float(match.group(1))
        else:
            result = None

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    result = run_remote_script()

    print(f"\nvalue: {result}")