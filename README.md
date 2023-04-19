# House-Rent-Prediction

This project aims to predict the rent of a house based on various features such as location, Furnishing Status, square footage, etc. The model has been trained on a dataset consisting of historical rental prices for houses in various Indian cities.


## File Structure

- `Data`: This folder contains the dataset used for training the model.
- `src`: This folder contains the source code for the project.
- `static`: This folder contains static files used by the web app.
- `templates`: This folder contains HTML templates used by the web app.
- `.gitignore`: This file specifies the files and folders that should be ignored by Git.
- `LICENSE`: This file contains the license information for the project.
- `README.md`: This file contains the documentation for the project.
- `app.py`: This file contains the code for the Flask web app.
- `requirements.txt`: This file contains the list of Python dependencies required to run the project.
- `model_trainer.py`: This file should be run before app.py, as it creates the preprocessing object and trains the machine learning model.
- `setup.py`: This file contains information about the project, such as its name, version, and dependencies.


## Usage
1. Clone this repository to your local machine.
2. Install the required dependencies by running pip install -r requirements.txt.
3. Run model_trainer.py to preprocess the data and train the machine learning model.
4. Run app.py to start the Flask web app.
5. Open a web browser and navigate to http://localhost:5000.
6. Enter the required details such as location, number of rooms, and square footage to get a predicted rent for the house

> `NOTE:` Make sure to run model_trainer.py before running app.py to ensure that the machine learning model has been trained and is ready for use in the web app.