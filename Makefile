ENV_NAME := nlp-intro
ifeq ($(OS),Windows_NT)
    ACTIVATE := activate
else
	ACTIVATE := . $(HOME)/anaconda3/bin/activate
endif


update-env:
	conda env update -n $(ENV_NAME) -f environment.yml


install-jupyterlab-ext:
	$(ACTIVATE) $(ENV_NAME) && \
	jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
	jupyter labextension install jupyterlab-plotly --no-build && \
	jupyter labextension install plotlywidget --no-build && \
	jupyter lab build


# Install python windows package manager to install wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/
install-windows-polyglot-dependencies:
ifeq ($(OS),Windows_NT)
	$(ACTIVATE) $(ENV_NAME) && pip install pipwin
	$(ACTIVATE) $(ENV_NAME) && pipwin install pyicu && pipwin install pycld2
endif


env: update-env install-windows-polyglot-dependencies install-jupyterlab-ext


remove-env:
	conda remove -n $(ENV_NAME) --all


lab:
	$(ACTIVATE) $(ENV_NAME) && jupyter lab


notebooks:
	$(ACTIVATE) $(ENV_NAME) && jupyter notebooks