ENV_NAME := nlp-intro


update-env:
	conda env update -n $(ENV_NAME) -f environment.yml && \
	. $(HOME)/anaconda3/bin/activate $(ENV_NAME) && \
	jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
	jupyter labextension install jupyterlab-plotly --no-build && \
	jupyter labextension install plotlywidget --no-build && \
	jupyter lab build

lab:
	. $(HOME)/anaconda3/bin/activate $(ENV_NAME) && \
	jupyter lab

remove-ipykernel:
	jupyter kernelspec uninstall $(ENV_NAME)

echo-usr:
	. $(HOME)/anaconda3/bin/activate $(ENV_NAME) && \
	conda list