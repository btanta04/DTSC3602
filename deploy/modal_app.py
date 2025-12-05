# modal_app.py
import shlex
import subprocess
from pathlib import Path

import modal

app = modal.App(name="outseer-streamlit-app")

BASE_DIR = Path(__file__).parent

# Local paths (on your laptop)
streamlit_local = BASE_DIR / "finalstreamlit.py"
articles_local = BASE_DIR / "outseer_articles.csv"
embeddings_local = BASE_DIR / "embeddings"
keywords_local = BASE_DIR / "bertopic_keywords_weights.csv"

# Sanity checks so you get a clear error if something is missing locally
for p in [streamlit_local, articles_local, embeddings_local, keywords_local]:
    if not p.exists():
        raise RuntimeError(f"Missing required file or folder: {p}")

streamlit_script_remote_path = "/root/finalstreamlit.py"

# Build image – only the libs this app actually uses
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit",
        "pandas",
        "numpy",
        "altair",
        "pyarrow",   # for read_parquet
    )
    .add_local_file(streamlit_local, streamlit_script_remote_path)
    .add_local_file(articles_local, "/root/outseer_articles.csv")
    .add_local_dir(embeddings_local, "/root/embeddings")
    .add_local_file(keywords_local, "/root/bertopic_keywords_weights.csv")
)

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def serve():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = (
        f"streamlit run {target} "
        f"--server.port 8000 "
        f"--server.address 0.0.0.0 "
        f"--server.enableCORS=false "
        f"--server.enableXsrfProtection=false"
    )
    # Use Popen, just like Modal's own example
    subprocess.Popen(cmd, shell=True)