# My PhD research

This project is an attempt to structure my work properly and avoid spaghetti code.
I've used Jupyter Notebooks exclusively for machine learning, mostly because it
"just works" and I haven't invested enough time to improve. However, I quickly
realised that my deep learning development is nowhere near the standards I have
when developing other projects, for example in C#.

## Re-creating environment

`requirements.txt` file contains a list of dependencies. It was exported using
`conda list -e > requirements.txt` command. To create new conda environment based on it
use `conda create --name <env> --file requirements.txt` command.

To create pip environment (not tested) use:
```
conda activate <env>
conda install pip
pip freeze > requirements.txt`
```

and then
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

