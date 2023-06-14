**Virat Kohli Performance Prediction Model**

This repository contains Python code to predict the performance of cricketer Virat Kohli using regression models. The code is divided into two parts:

   Virat-Kohli-Performance-Prediction-Model: Predicts the runs scored by Virat Kohli in an ODI match based on different factors such as balls faced, minutes batted, and        boundaries hit.

   Match-Outcome-Prediction: Predicts whether the match result for a match involving Virat Kohli will be a win or a loss based on his performance statistics.

**Dataset**

   The code uses the "Virat_Kohli_odi.csv" dataset, which contains the ODI statistics of Virat Kohli. The dataset includes the following columns:

    Runs Scored: The total runs scored by Virat Kohli in each ODI match.
    Minutes Batted: The number of minutes Virat Kohli batted in each match.
    Balls Faced: The number of balls faced by Virat Kohli in each match.
    Boundaries: The number of boundaries hit by Virat Kohli in each match.
    Strike Rate: The strike rate of Virat Kohli in each match.
    Result: The match result ('w' for win and 'l' for loss).
    

**Dependencies**

   The following Python libraries are required to run the code:

    pandas: Data manipulation and analysis library.
    scikit-learn: Machine learning library for regression and classification models.
    plotnine: Data visualization library based on ggplot2.
    matplotlib: Data visualization library for creating plots.
    
   Install the dependencies using the following command:

    pip install pandas scikit-learn plotnine matplotlib

**Usage**

   Ensure you have Python 3.x installed on your machine.

   Clone the repository:

    git clone https://github.com/azafar224/Virat-Kohli-Performance-Prediction-Model.git
    
   Navigate to the project directory:

    cd Virat-Kohli-Performance-Prediction-Model

   Place the "Virat_Kohli_odi.csv" dataset file in the project directory.

   Run the desired code:

   To predict Virat Kohli's performance, run:

    python Kohli_regression_analysis.py
    
   To predict the match outcome, run:

    python  Kohli_logistic_regression_analysis.py
   
   The code will load the dataset, preprocess the data, train the regression or classification models, and provide the predictions or match outcomes.

**Results**

 Kohli_Regression_Analysis: The code will display statistical measures for each numeric column, R^2 scores for the regression models, and the predicted runs for a given input. It will also generate scatter plots to visualize the relationships between runs scored and balls faced, as well as runs scored and minutes batted.

 Kohli_Logistic_Regression_Analysis: The code will predict whether the match result for a match involving Virat Kohli will be a win or a loss based on his performance statistics. It will print "Win" or "Loss" for each match outcome and display the accuracy score of the logistic regression model.

**Contributing**

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

**Acknowledgements**

The dataset used in this analysis is sourced from official cricket records. We would like to express our gratitude to the cricket community and statisticians for providing the data.

Contact
For any questions or inquiries, please contact [ahmadzafar224@gmail.com].

Thank you for using the Kohli Prediction script. May your cricket analytics journey be fruitful and insightful!
