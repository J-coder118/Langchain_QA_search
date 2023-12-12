## How to execute this project on your machine ##

(I expected you have already installed python and pip)
1. load this project into vscode.
2. Set up the virtual environment.
# Create a virtual environment to isolate our package dependencies locally
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
3. Installing packages.
python3 install -r requirements.txt
then 
pip3 install chromadb pandas streamlit sentence_transformers
then
streamlit run main.py

you can test with this demo interface by clicking answer button 2 times.