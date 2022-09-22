# Shiny GA feature selection


## Installation:

### First clone this repository:

    `git clone https://github.com/amgfernandes/shiny_GA_feature_selection.git`

### Then move inside the folder created:

    `cd shiny_GA_feature_selection`

### Create a new environment to install the app

- Example with new environment named `shiny_GA_feature_selection`

```
conda create -n shiny_GA_feature_selection python=3.9 -y

conda activate shiny_GA_feature_selection

conda install pip -y

pip install -r requirements.txt
 ```

### You can then run the Shiny app with the following:

`shiny run --reload myapp/app.py`

-  open a browser with following link: http://localhost:8000:

<br>

<img src="img/shiny_browser.png" alt="Screenshot_browser" width="1200"/>

<br>

- or run inside VS Code:
<br>

<img src="img/shiny_vscode.png" alt="Screenshot_browser" width="1200"/>
