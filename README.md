# Building NLP based ML-model on Kaggle's Question-pairs dataset using FastAPI and Docker

The requirement of this project is to predict which of the provided pairs of questions contain two questions with the same meaning. <br>
***Sample*** <br>
question1 - 'How can I be a good geologist?' <br>
question2 - 'What should I do to be a great geologist?' 	

You can download the dataset here [Question pairs dataset | Kaggle](https://www.kaggle.com/competitions/quora-question-pairs/data)

Download the train.csv and test.csv and keep it in input folder. To train the dataset or to batch predict using csv file run "main.py". <br>
To run the app server using Fast API follow below documentation

## Steps to run the server:


#### Build and run docker container <br>
```
    docker-compose up -d --build 
```
The above command builds the docker image and runs it.
Check if docker container "qp_similarity_container" is running using ```docker ps``` <br>
We can use Postman to test the api http://0.0.0.0:8212/qp_similarity by passing the json in the body. It returns the json (eg. {"is_duplicate": "1"}) which indicates whether the questions are duplicate or not. 

Alternatively, open the browser and type url as http://0.0.0.0:8212/docs (FastAPI swagger). In the POST ../qp_similarity request body paste the json from sample.json file and click on execute. 

Now the docker container can be deployed anywhere.

## Deployment of the solution
#### Following are the few methods we can use to deploy the solution

1. Manual deployment to server instance: <br>
    * After building the docker container, push the container to private docker registry.
    * Now at the server instance we can pull our docker image from our private docker registry using docker credentials and run the container. 

1. Deployment using Google cloud App Engine (Serverless)
    * Create app.yaml an configuration file that basically contains app engine settings.
    * By runnning ```gcloud app deploy``` , it pushes all files from the current directory to the cloud. Further, it builds the docker image in the cloud and stores it in Google artifact registry. It uses this docker image to deploy the container (running instance of the image) on the App Engine. 
    
1. Deployment using Google compute engine: <br>
    * Once we build the docker container, push the docker image to Google cloud Artifact registry using gcloud 
    credentials. <br>
    * Create compute instance (VM) in Google cloud platform. Pull the docker image from the Artifact registry and run the container.
    
1. Deployment using Kubernetes
    * Install Kubernetes or use Google Kuberentes Engine if we need to use other cloud functionalities. 
    * Create deployment.yaml file to configure the replicas, container, label etc. and create service.yaml to communicate with pods.
    * Once we have the docker image built and is pushed to docker registry server, the docker credentials having our pushed image which will get deployed to kubernetes after running the deployment.yaml and service.yaml using kubectl.
    
    
    
## Monitoring:

* We can monitor on a daily or weekly interval by running script or with a cron job to check whether the number of bot transaction over the interval has exceeded the threshold. This implies either our model is not performing well and needs to retrain or there is a suspicious activity occuring too frequently.

* Similarly we can also check for the overall latency across the interval and if the latency is not below 200ms then trigger the retraining script with new data or send an alert to retrain

* In deployment using Google compute engine we can monitor using cloud scheduler with cloud functions where we can add functionalities to check latency constraint, get-new-data, retrain, etc. Cloud scheduler can schedule jobs in realtime, daily or weekly interval on these cloud functions.

* Deploying model on kubernetes solves the problem of scaling and server failure of our application. If we are using Google Kubernetes Engine we can leverage cloud scheduler and cloud functions to monitor the application.


## Production:

* The above mentioned all deployment methods can be used for production depending on our requirements.

* If we need to quickly deploy the solution at scale, we can deploy using Google App Engine which is serverless that manages the infrastracture like scaling (increase in the nodes based on traffic).

* If we are low on budget than we can deploy in one our server instances. But it is not easy to scale.

* Deployment in Google compute engine is less expensive and the monitoring is can be done using cloud schedular and cloud function which is easy build. Scaling cannot be done realtime but can scale in regular interval by changing the configuration.

* Deployment on orchestrating tool like kubernetes with kubeflow pipeline can automate all the above tasks. Easy to scale, monitoring and an intuitive dashboard to check the metrics, logs, latency, all at on place. Kubeflow seems to good solution which comprises of model development, deployment and monitoring.  




## Unit Test

* Install pytest using ```pip install pytest```
* Run all the unit test cases using below command: <br>
```pytest tests/unit_test```
