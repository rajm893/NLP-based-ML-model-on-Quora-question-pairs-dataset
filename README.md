# Building NLP based ML-model on Kaggle's Question-pairs dataset using FastAPI and Docker 📚🔍

## Project Overview 📋

This project involves the development of a Natural Language Processing (NLP) based machine learning model for predicting whether a pair of questions have the same meaning. The dataset used for this project is Kaggle's Question-pairs dataset, and the implementation is done using FastAPI and Docker.

### Sample Data 📝
Sample question pair from the dataset:
- **Question 1:** 'How can I be a good geologist?'
- **Question 2:** 'What should I do to be a great geologist?'

You can download the dataset [here](https://www.kaggle.com/c/quora-question-pairs/data) (Question pairs dataset | Kaggle). Download the `train.csv` and `test.csv` files and place them in the `input` folder.

### Running the Project ▶️

To train the dataset or batch predict using a CSV file, execute the following command:
```
python main.py
```
For running the app server using FastAPI, follow the provided documentation.

### Steps to Run the Server 🐳:
Build and run the Docker container:

```
docker-compose up -d --build
```

Verify if the Docker container named "qp_similarity_container" is running using:
```
docker ps
```

#### Testing the API 🚀:
You can use Postman to test the API at `http://0.0.0.0:8212/qp_similarity` by passing JSON data in the request body. It returns JSON indicating whether the questions are duplicate or not (e.g., `{"is_duplicate": "1"}`).
Alternatively, open a web browser and access `http://0.0.0.0:8212/docs` (FastAPI swagger). Paste the JSON data from the `sample.json` file into the request body and click "execute."

### Deployment Options

🖥️ **Manual Deployment to Server Instance** :
- After building the Docker container, push it to a private Docker registry. At the server instance, pull the Docker image from the private registry using Docker credentials and run the container.

☁️ **Deployment using Google Cloud App Engine (Serverless)**:
- Create an `app.yaml` configuration file with App Engine settings. Deploy the project using `gcloud app deploy`, which pushes files to the cloud, builds a Docker image in the cloud, and deploys the container to App Engine.

🗃️ **Deployment using Google Compute Engine**:
- Build the Docker container and push it to Google Cloud Artifact Registry using `gcloud` credentials. Create a compute instance (VM) on Google Cloud Platform, pull the Docker image from the Artifact Registry, and run the container.

 ☸️ **Deployment using Kubernetes**:
- Install Kubernetes or use Google Kubernetes Engine for other cloud functionalities. Create `deployment.yaml` to configure replicas, containers, labels, etc., and `service.yaml` for pod communication. After pushing the Docker image to the registry server, deploy the container to Kubernetes using `kubectl`.

### Monitoring and Production 📊🚀

**Monitoring**:
- Implement monitoring on a daily or weekly basis to check whether the number of duplicate questions exceeds a certain threshold. This can indicate model performance or increased duplicate questions.

**Production**:
- The choice of deployment method depends on project requirements:
   - Google App Engine for rapid scaling and serverless management.
   - Server instances for budget-conscious projects.=p
   - Google Compute Engine for cost-effective deployment with monitoring using cloud scheduler and cloud functions.
   - Kubernetes with Kubeflow for an automated, scalable, and monitored solution.
   
### Unit Testing 🧪
- To run unit tests, install pytest using `pip install pytest`.
- Execute all unit test cases with the following command:
`pytest tests/unit_test`
