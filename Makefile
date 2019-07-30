ENV_NAME := nlp-intro
ifeq ($(OS),Windows_NT)
    ACTIVATE := activate
    INSTALL_SHAPELY :=
else
	ACTIVATE := . $(HOME)/anaconda3/bin/activate
endif

update-env:
	conda env update -n $(ENV_NAME) -f environment.yml && \
	$(ACTIVATE) $(ENV_NAME) && \
	jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
	jupyter labextension install jupyterlab-plotly --no-build && \
	jupyter labextension install plotlywidget --no-build && \
	jupyter lab build

lab:
	$(ACTIVATE) $(ENV_NAME) && jupyter lab
