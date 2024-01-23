from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from helper_utils import write_to_csv
import pandas as pd
import preprocessing
import unsupervised_clustering

load_dotenv()
app = FastAPI()


class Project(BaseModel):
    project_name: str
    client_name: str
    domain_name: str


class ClusterDetail(BaseModel):
    no_of_cluster: int
    feature_1: str
    feature_2: str
    feature_3: str
    score_1: float
    score_2: float
    score_3: float


class SkillDetail(BaseModel):
    python: int = Field(default=0)
    machine_learning: int = Field(default=0)
    deep_learning: int = Field(default=0)
    data_analysis: int = Field(default=0)
    asp_net: int = Field(default=0)
    ado_net: int = Field(default=0)
    vb_net: int = Field(default=0)
    csharp: int = Field(default=0)
    java: int = Field(default=0)
    spring_boot: int = Field(default=0)
    hibernate: int = Field(default=0)
    nlp: int = Field(default=0)
    cv: int = Field(default=0)
    js: int = Field(default=0)
    react: int = Field(default=0)
    node: int = Field(default=0)
    angular: int = Field(default=0)
    dart: int = Field(default=0)
    flutter: int = Field(default=0)


@app.post("/api/project-details/")
async def add_project_info(project: Project):
    if not (project.project_name and project.client_name and project.domain_name):
        raise HTTPException(status_code=400, detail="Enter all the details")
    file_path = os.getenv('PROJECT_DETAILS')
    write_to_csv(file_path, [project.project_name,
                 project.client_name, project.domain_name], header=["Project Name", "Client Name", "Domain Name"])
    return {"Sucess": True}


@app.post("/api/cluster-detail/")
async def add_cluster_detail(cluster_detail: ClusterDetail):
    if not (cluster_detail.no_of_cluster and cluster_detail.feature_1 and cluster_detail.feature_2
            and cluster_detail.feature_3 and cluster_detail.score_1 and cluster_detail.score_2
            and cluster_detail.score_3):
        raise HTTPException(status_code=400, detail="Enter all the details")
    file_path = os.getenv('CLUSTER_DETAILS')

    write_to_csv(file_path, [cluster_detail.no_of_cluster, cluster_detail.feature_1,
                             cluster_detail.feature_2, cluster_detail.feature_3,
                             cluster_detail.score_1, cluster_detail.score_2, cluster_detail.score_3], header=["No. of Cluster", "Feature 1", "Feature 2", "Feature 3", "Score 1", "Score 2", "Score 3"])

    return {"success": True}


@app.post("/api/skill-detail/")
async def add_skill_detail(skill_detail: SkillDetail):
    fields = [skill_detail.python, skill_detail.machine_learning, skill_detail.deep_learning,
              skill_detail.data_analysis, skill_detail.asp_net, skill_detail.ado_net,
              skill_detail.vb_net, skill_detail.csharp, skill_detail.java,
              skill_detail.spring_boot, skill_detail.hibernate, skill_detail.nlp,
              skill_detail.cv, skill_detail.js, skill_detail.react,
              skill_detail.node, skill_detail.angular, skill_detail.dart, skill_detail.flutter]

    header = ["Python", "Machine Learning", "Deep Learning", "Data Analysis",
              "ASP.Net", "ADO.Net", "VB.Net", "C#", "Java",
              "Spring Boot", "Hibernate", "NLP", "CV", "JS", "React",
              "Node", "Angular", "Dart", "Flutter"]
    file_path = os.getenv('SKILL_DETAILS')
    write_to_csv(file_path, fields, header)
    return {"success": True}


@app.post("/api/content-detail/")
async def add_content_detail(feature_1: str, feature_2: str, feature_3: str, feature_4: str):
    if any(field is None or field.strip() == '' for field in [feature_1, feature_2, feature_3, feature_4]):
        raise HTTPException(status_code=400, detail="All fields are required")
    file_path = os.getenv('FINAL_DETAILS')
    fields = [feature_1, feature_2, feature_3, feature_4]
    header = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    write_to_csv(file_path, fields, header)
    return {"success": True}


async def perform_clustering():
    file_path = os.getenv('CLUSTERING_DATA')
    data = pd.read_csv(file_path)
    dataset = preprocessing.DataSet(data)
    cluster_details = pd.read_csv(os.getenv['CLUSTER_DETAILS'])
    cluster_model = unsupervised_clustering.UnsupervisedClustering(
        dataset.data)
