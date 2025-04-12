<h1 align="center">HealthEdgeAI: GAI and XAI Based Healthcare System for Sustainable Edge AI and Cloud Computing Environments</h1>

## Dataset
The dataset used in this project is based on the Statlog dataset, which merges five heart disease datasets from different sources. You can download the dataset from the following link:

- [Download Dataset](https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive)

The dataset included in this repository consists of both the original and synthetically generated data using Generative AI:
- `1190_ieee_heart_statlog_cleveland_hungary_final.csv` - This is the original dataset.
- `heart_gen_ai_917_source.csv` - This is the pre-processed original dataset acting as the source for the Generative AI model API.
- `syn_data_917_5000.csv` - This is the resulting synthetic dataset containing 5000 records.
- `syn_data_917_10000.csv` - This is the resulting synthetic dataset containing 10000 records.

## Notebook Overview
- **EDA_Model_Training.ipynb** - This Jupyter notebook contains the code for all the exploratory data analysis (EDA), model training, evaluation, and saving the model.
- **QoS_Comparison.ipynb**- This notebook loads all the JMeter test outputs and converts them into suitable comparison plots for analyzing the QoS metrics.
- **Synthetic_Data_Generation.ipynb** - This notebook contains the code for loading the pre-processed datasets and generating synthetic datasets using the Generative AI model.

## Steps to Run the Code

Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Running the Streamlit Application
1. Navigate to the project directory.
2. Run the Streamlit app using the following command:
   ```bash
   streamlit run app.py
   ```
3. The application will be available at `http://localhost:8501/`.

### Running the FastAPI Application
1. Navigate to the project directory.
2. Run the FastAPI application using the following command:
   ```bash
   uvicorn app:app --reload
   ```
3. The FastAPI application will be available at `http://localhost:8000/`.
4. You can access the automatically generated API documentation at `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc.

### Additional Scripts
- To generate predictions, use the `prediction.py` script:
  ```bash
  python prediction.py
  ```
- For edge and cloud deployment setup, refer to the `setup.sh` script.

## Deploying the Application on Microsoft Azure

1. **Navigate to the Azure Portal:**
   - Go to the [Azure Portal](https://portal.azure.com/).
2. **Create an Azure Web App:**
   - In the Azure Portal, search for "App Services" and click on "Create."
   - Fill in the necessary details such as Subscription, Resource Group, and Web App Name.
3. **Configure Deployment Settings:**
   - Under the "Deployment" section, choose "GitHub" as your deployment source.
   - Authenticate with your GitHub account if you haven’t done so already.
   - Select your repository and the branch you want to deploy (e.g., `main`).
4. **Set Up Continuous Integration/Continuous Deployment (CI/CD):**
   - In the deployment settings, ensure that the "Enable Continuous Deployment" option is selected. This will automatically trigger deployments whenever you push changes to your GitHub repository.
5. **Review and Create:**
   - Review your settings and click "Create" to deploy your application.
6. **Access Your Deployed Application:**
   - Once the deployment is complete, you can access your Streamlit application via the provided Azure Web App URL (e.g., `https://your-app-name.azurewebsites.net/`).
7. **Monitoring and Scaling:**
   - Use the Azure Portal to monitor the performance of your application. Azure provides built-in tools to scale your app based on demand.

## JMeter Performance Testing


1. **Install Apache JMeter:**
   - Download and install Apache JMeter from [the official website](https://jmeter.apache.org/).

2. **Configure Test Plan:**
   - Open JMeter and create a new Test Plan named `your-test-plan-name`.

3. **Adding HTTP Requests:**
   - Add a thread group for each test scenario (e.g., Edge Device - Asus ROG Laptop, Edge Device - Alienware Laptop, Azure Web Service).
   - Inside each thread group, add an `HTTP Request` sampler

4. **Configure HTTP Requests:**
   - For each device or service, set up the `HTTP Request` as follows:
     - **Protocol**: `http`
     - **Server Name or IP**: (use the respective IP address or domain, but do not include it in public documents)
     - **Port Number**: `8001` (or the appropriate port for your service)
     - **Path**: `/predict`
     - **Method**: `POST`
     - **Body Data**: Include the JSON payload for the prediction.

5. **Adding Listeners:**
   - Add listeners such as `View Results Tree`, `View Results in Table`, and `Summary Report` to capture and analyze the results.

6. **Run the Test:**
   - Execute the test plan and compare the results across different devices and deployment scenarios.

## Cite this work
Our work is published in Concurrency and Computation: Practice and Experience, cite using the following bibtex entry:
```bibtex
@article{HealthEdgeAI,
author = {Wang, Han and Chelvan, Balaji Muthurathinam Panneer and Golec, Muhammed and Gill, Sukhpal Singh and Uhlig, Steve},
title = {HealthEdgeAI: GAI and XAI Based Healthcare System for Sustainable Edge AI and Cloud Computing Environments},
journal = {Concurrency and Computation: Practice and Experience},
volume = {37},
number = {9-11},
pages = {e70057},
doi = {https://doi.org/10.1002/cpe.70057},
year = {2025}}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.