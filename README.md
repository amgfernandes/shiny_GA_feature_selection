# Shiny GA feature selection


### Install

First clone this repository:

`git clone https://github.com/amgfernandes/shiny_GA_feature_selection.git`

then go inside the folder created:

`cd shiny_GA_feature_selection`

Example with new environment named `shiny`

```
conda create -n shiny python=3.9 -y

conda activate shiny

conda install pip -y

pip install -r requirements.txt
 ```

You can then run shiny lik:

`shiny run --reload app.py`

-  open a browser: http://localhost:8000:

<img src="img/shiny_browser.png" alt="Screenshot_browser" width="1200"/>

- or run inside VS Code:
<img src="img/shiny_vscode.png" alt="Screenshot_browser" width="1200"/>
