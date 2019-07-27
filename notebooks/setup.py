import sys
from pathlib import Path
from plotly.offline import init_notebook_mode

# Add this project to the path
project_path = str(Path(__file__).parents[1].resolve())
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Initialize notebook mode for plotly
init_notebook_mode(connected=True)
