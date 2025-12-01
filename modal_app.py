# modal_app.py
import shlex
import subprocess
from pathlib import Path

import modal

# Name of your Modal app
app = modal.App(name="streamlit-outseer-deployment")

# Paths to your local files
streamlit_script_local_path = Path(__file__).parent / "finalstreamlit.py"
streamlit_script_remote_path = "/root/finalstreamlit.py"

csv_local_path = Path(__file__).parent / "outseer_articles.csv"
csv_remote_path = "/root/outseer_articles.csv"

# Define image: install deps + copy in your files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
    .add_local_file(csv_local_path, csv_remote_path)
)

# Attach the image to the app
app.image = image

# Optional safety: fail fast if the Streamlit script is missing locally
if not streamlit_script_local_path.exists():
    raise RuntimeError("finalstreamlit.py not found in the same directory as modal_app.py")

if not csv_local_path.exists():
    raise RuntimeError("outseer_articles.csv not found in the same directory as modal_app.py")


@app.function()
@modal.web_server(8501)
def serve():
    # Run Streamlit in a background process
    cmd = f"streamlit run {shlex.quote(streamlit_script_remote_path)} " \
          f"--server.port 8501 " \
          f"--server.address 0.0.0.0 " \
          f"--server.enableCORS=false " \
          f"--server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)