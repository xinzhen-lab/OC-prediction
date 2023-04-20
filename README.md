# Early-stage Ovarian Cancer Prediction
This tool is developed and validated based on the data of three cohorts (with 2000+ positive cases vs. 10000+ controls) from hospitals in China: Sun Yat-sen University Cancer Center (SYSUCC), Tongji Hospital of Huazhong University of Science & Technology (TJH) and Women's Hospital of Zhejiang University (WHZJU). The internally validated AUC (TJH cohort) was 0.951 while the AUCs on the independent cohorts (SYSUCC & WHZJU) were respectively 0.885 and 0.882, indicating good discriminative ability. This model was developed based a multi-classifiers fusion methodology [1]. Other investigators are invited to share their data in order to further improve the generalization capability of the current model. The model should only be used to support clinical judgment by health care professionals as a complementary tool for screening early-stage ovarian cancer (OC) with laboratory tests. Any responsibility for using this model and its results will rest solely by the health care professional using the model. Using it you should understand and agree that this site is not responsible or liable for any claim, loss, or damage resulting from its use. While we try to keep the information on the site as accurate as possible, we disclaim any warranty concerning its accuracy, timeliness, and completeness, and any other warranty, express or implied, including warranties of merchantability or fitness for a particular purpose.

# Reference
[1] Q He, X Li, DWN Kim, X Jia, X Gu, X Zhen, L Zhou: Feasibility study of a multi-criteria decision-making based hierarchical model for multi-modality feature and multi-classifier fusion: Applications in medical prognosis prediction. Information Fusion 2020, 55:207-219.(https://doi.org/10.1016/j.inffus.2019.09.001)

# Note
Missing values please leave blank. This model is tolerant of missing values. Note that more accurate predictions might be expected only when sufficient lab test items are provided.

# Screenshot
![image](https://user-images.githubusercontent.com/131331281/233329985-1f32a9ab-242a-4ca2-b1aa-f02d3cff00a2.png)

# Requirements
python-3.7.12
Flask==1.1.1
gunicorn==20.0.4
itsdangerous==1.1.0
Jinja2==2.11.1
joblib==0.16.0
MarkupSafe==1.1.1
Werkzeug==0.16.1
setuptools==57.0.0
setuptools-scm==6.3.2	
six==1.16.0	
sklearn==0.0	
tenacity==8.0.1	
threadpoolctl==3.0.0	
tomli==1.2.2	
typing-extensions==4.0.1	
wheel==0.36.2	
xgboost==0.90	
zipp==3.6.0
plotly==5.4.0	
pyparsing==3.0.6
python-dateutil==2.8.2	
pytz==2021.3
packaging==21.3
kiwisolver==1.3.2	
lightgbm==2.3.0
catboost==0.26	
click==8.0.3	
colorama==0.4.4	
cycler==0.11.0	
fonttools==4.28.3	
graphviz==0.19	
importlib-metadata==4.8.2	
Pillow==8.4.0

# Usage
run app.py, then click the http://127.0.0.1:5000/ link on the output window to launch the tool on your local browser.
![image](https://user-images.githubusercontent.com/131331281/233333144-6dcc19c0-9c8a-4b1f-b8db-9657c23d687e.png)
