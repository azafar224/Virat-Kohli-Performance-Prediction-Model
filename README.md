This Python script analyzes a dataset containing the ODI (One Day International) cricket statistics of Virat Kohli, one of the finest batsmen in the world. It explores various regression techniques to predict the runs scored by Virat Kohli based on different factors such as balls faced, minutes batted, and boundaries hit.

Dataset
The dataset used for this analysis is stored in the file "Virat_Kohli_odi.csv". It contains the following columns:

Runs Scored: The total runs scored by Virat Kohli in each ODI match.
Minutes Batted: The number of minutes Virat Kohli batted in each match.
Balls Faced: The number of balls faced by Virat Kohli in each match.
Boundaries: The number of boundaries hit by Virat Kohli in each match.
Strike Rate: The strike rate of Virat Kohli in each match.


Features
Loading and preprocessing the dataset.
Handling missing values by removing rows with missing values.
Calculating statistical measures such as mean, median, mode, standard deviation, and variance for each numeric column.
Performing simple linear regression to predict runs based on balls faced.
Performing multiple linear regression to predict runs based on minutes batted, balls faced, and boundaries.
Applying polynomial regression to capture nonlinear relationships between the independent and dependent variables.
Splitting the dataset into training and testing sets for model evaluation.
Evaluating the performance of each regression model using the coefficient of determination (R^2 score).
Predicting the runs scored by Virat Kohli using the trained regression models.
Visualizing the relationship between runs scored and balls faced, as well as runs scored and minutes batted using scatter plots.
Dependencies
The following Python libraries are required to run the script:

pandas: Data manipulation and analysis library.
scikit-learn: Machine learning library for regression models.
plotnine: Data visualization library based on ggplot2.
Install the dependencies using the following command:

bash
Copy code
pip install pandas scikit-learn plotnine
Usage
Ensure you have Python 3.x installed on your machine.

Clone the repository:

bash
Copy code
git clone https://github.com/your_username/cricket-runs-prediction.git
Navigate to the project directory:

bash
Copy code
cd cricket-runs-prediction
Place the "Virat_Kohli_odi.csv" dataset file in the project directory.

Run the script:

    python cricket_runs_prediction.py

The script will load and preprocess the dataset, perform regression analysis using different models, evaluate their performance, and generate scatter plots for visualization.

Results
The script will display the statistical measures for each numeric column, the R^2 scores for the regression models, and the predicted runs for a given input. Additionally, two scatter plots will be generated to visualize the relationships between runs scored and balls faced, as well as runs scored and minutes batted.

Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
The dataset used in this analysis is sourced from official cricket records. We would like to express our gratitude to the cricket community and statisticians for providing the data.

Contact
For any questions or inquiries, please contact [your_email@example.com].

Thank you for using the Cricket Runs Prediction script. May your cricket analytics journey be fruitful and insightful!
