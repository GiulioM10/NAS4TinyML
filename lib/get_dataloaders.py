import sys
import subprocess

subprocess.check_call([sys.executable, 'bash', 'import_dataset.sh'])
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pyvww'])