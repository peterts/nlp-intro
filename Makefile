ENV_NAME := nlp-intro
ifeq ($(OS),Windows_NT)
    ACTIVATE := activate
else
	ACTIVATE := . $(HOME)/anaconda3/bin/activate
	# Darwin = OSX
	UNAME_S := $(shell uname -s)
endif


update-env:
	conda env update -n $(ENV_NAME) -f environment.yml


install-jupyterlab-ext:
	$(ACTIVATE) $(ENV_NAME) && \
	jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
	jupyter labextension install jupyterlab-plotly --no-build && \
	jupyter labextension install plotlywidget --no-build && \
	jupyter labextension install @jupyterlab/toc --no-build && \
	jupyter lab build


# Install python windows package manager to install wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/
install-polyglot:
ifeq ($(OS),Windows_NT)
	$(ACTIVATE) $(ENV_NAME) && pip install pipwin
	$(ACTIVATE) $(ENV_NAME) && pipwin install pyicu && pipwin install pycld2
else ifeq ($(UNAME_S),Darwin) # OSX
	brew install icu4c
	$(ACTIVATE) && pip install pyicu
endif
	$(ACTIVATE) && pip install git+https://github.com/aboSamoor/polyglot.git


env: update-env install-jupyterlab-ext


remove-env:
	conda remove -n $(ENV_NAME) --all


lab:
	$(ACTIVATE) $(ENV_NAME) && jupyter lab


notebooks:
	$(ACTIVATE) $(ENV_NAME) && jupyter notebooks


get-data:
	$(ACTIVATE) && python nlp_intro/diskusjon_no_scraper.py