ChatGPT3 Market Risk Analyzer

This project involves downloading, transforming, and filtering 9 datasets from the FRED API using Python, and analyzing their normal distribution with the Pandas library. An algorithm categorizes each data point based on historical normal distribution, and a ChatGPT3 model is fine-tuned using 40 years of CPI, PCE, and Inflation data.

Key user stories include:

-Entering data points into the database to create prompts for ChatGPT3.
-Viewing ChatGPT3 completions on a dashboard.
-Exploring and navigating datasets in the project.
-Assessing the market's risk level.
-Evaluating the accuracy rate of ChatGPT3's completion response.
-Retraining the model based on accuracy results.

Backend functionalities include:

-ChatGPT3 Model: Saving responses, retraining, and creating .jsonl files.
-Economic Data Model: Creating Plotly charts, fetching economic data, and retrieving the latest data and dates from FRED API.
-Results Model: Saving and retrieving statistical analysis data, risk analysis, and creating Plotly gauge charts.
-User Data Model: Saving and fetching the latest user data.
-User Model: Saving and verifying users, validating names.
-User Controller: Registration and login with Bcrypt and Regex, and clearing sessions.
-User Data Controller: Creating and processing prompts, retraining the model, and managing the dashboard.

Next Steps include deploying to AWS and refining the documentation to aid other developers. In the mean time. Feel free to Connect a MySQL database, Install Requirements, and Run with "python server.py"
