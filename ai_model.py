from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None
location_encoder = None
availability_encoder = None
scaler = None
jobs = None

def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    skills = [
        'python', 'java', 'javascript', 'html', 'css', 'r', 'sql', 'machine_learning',
        'data_analysis', 'web_development', 'technical_support', 'ui_ux_design',
        'database_management', 'testing', 'linux', 'windows', 'troubleshooting',
        'aws', 'networking', 'product_management', 'unity', 'business_analysis',
        'digital_marketing', 'technical_writing', 'salesforce', 'docker', 'azure',
        'bi_tools', 'systems_analysis', 'cloud_computing', 'cyber_security', 'devops'
    ]
    locations = [
        'New York', 'San Francisco', 'Remote', 'Chicago', 'Boston', 'Austin',
        'Seattle', 'Los Angeles', 'San Diego', 'Philadelphia', 'Dallas', 'Denver',
        'Washington DC', 'Atlanta', 'Miami', 'Houston', 'San Jose', 'Cleveland',
        'Orlando', 'Charlotte', 'Minneapolis', 'Detroit', 'Indianapolis', 'Pittsburgh',
        'St. Louis', 'Baltimore', 'Nashville', 'Kansas City', 'Columbus', 'Salt Lake City',
        'Portland', 'San Antonio', 'Sacramento', 'San Bernardino', 'Cincinnati',
        'Jacksonville', 'Tampa', 'Raleigh', 'Omaha', 'Louisville', 'Milwaukee',
        'Birmingham', 'Tulsa', 'Memphis', 'Richmond', 'Buffalo'
    ]
    availabilities = [
        'Immediately', '1-2 Weeks', '1 Month', 'Flexible', 'Not Available'
    ]

    data = []
    for _ in range(n_samples):
        job_skills = np.random.choice(skills, size=np.random.randint(2, 6), replace=False)
        experience = np.random.randint(0, 11)
        location = np.random.choice(locations)
        salary = np.random.randint(50000, 150001)
        availability = np.random.choice(availabilities)

        match = 1 if (len(job_skills) >= 3 and experience >= 0 and salary >= 80000) else 0

        data.append({
            'skills': ' '.join(job_skills),
            'experience': experience,
            'location': location,
            'salary': salary,
            'availability': availability,
            'match': match
        })

    return pd.DataFrame(data)

def prepare_features(df, vectorizer=None, location_encoder=None, availability_encoder=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        skills_vectorized = vectorizer.fit_transform(df['skills'])
    else:
        skills_vectorized = vectorizer.transform(df['skills'])

    if location_encoder is None:
        location_encoder = LabelEncoder()
        location_encoded = location_encoder.fit_transform(df['location'])
        location_encoded = pd.get_dummies(location_encoded, prefix='location')
    else:
        location_encoded = location_encoder.transform(df['location'])
        location_encoded = pd.get_dummies(location_encoded, prefix='location')

    if availability_encoder is None:
        availability_encoder = LabelEncoder()
        availability_encoded = availability_encoder.fit_transform(df['availability'])
        availability_encoded = pd.get_dummies(availability_encoded, prefix='availability')
    else:
        availability_encoded = availability_encoder.transform(df['availability'])
        availability_encoded = pd.get_dummies(availability_encoded, prefix='availability')

    features = np.hstack([
        skills_vectorized.toarray(),
        df[['experience', 'salary']].values,
        location_encoded.values,
        availability_encoded.values
    ])

    return features, vectorizer, location_encoder, availability_encoder


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    return model, scaler

def predict_job_matches(user_profile, jobs, model, vectorizer, location_encoder, availability_encoder, scaler):
    matched_jobs = []

    for job in jobs:
        # Check if the job experience meets the user's experience
        if job['experience'] > user_profile['experience']:
            continue

        combined_profile = {
            'skills': user_profile['skills'] + ' ' + job['skills'],
            'experience': job['experience'],
            'location': job['location'],
            'salary': job['salary'],
            'availability': job['availability']
        }

        combined_df = pd.DataFrame([combined_profile])

        features, _, _, _ = prepare_features(combined_df, vectorizer, location_encoder, availability_encoder)

        if features.shape[1] < scaler.n_features_in_:
            pad_width = ((0, 0), (0, scaler.n_features_in_ - features.shape[1]))
            features = np.pad(features, pad_width, mode='constant', constant_values=0)

        features_scaled = scaler.transform(features)
        match_probability = model.predict_proba(features_scaled)[0][1]
        if match_probability >= 0.80:
            matched_jobs.append({
                "job": job,
                "match_score": match_probability
            })

    return matched_jobs

@app.route('/api/train', methods=['GET'])
def train():
    global model, vectorizer, location_encoder, availability_encoder, scaler, jobs
    df = generate_dummy_data()
    X, vectorizer, location_encoder, availability_encoder = prepare_features(df)
    y = df['match'].values

    model, scaler = train_model(X, y)
    jobs = [
        {"id": 1, "title": "Backend Developer", "skills": "java sql web_development", "experience": 3, "location": "New York", "salary": 120000, "availability": "1 Month"},
        {"id": 2, "title": "Frontend Developer", "skills": "javascript html css", "experience": 2, "location": "San Francisco", "salary": 110000, "availability": "Flexible"},
        {"id": 3, "title": "Software Engineer", "skills": "java python web_development", "experience": 4, "location": "Seattle", "salary": 130000, "availability": "1-2 Weeks"},
        {"id": 4, "title": "Data Analyst", "skills": "r sql data_analysis", "experience": 3, "location": "Chicago", "salary": 100000, "availability": "Immediately"},
        {"id": 5, "title": "Web Developer", "skills": "javascript html css", "experience": 1, "location": "Boston", "salary": 95000, "availability": "Flexible"},
        {"id": 6, "title": "DevOps Engineer", "skills": "docker kubernetes aws", "experience": 5, "location": "Austin", "salary": 140000, "availability": "1 Month"},
        {"id": 7, "title": "Machine Learning Engineer", "skills": "python machine_learning deep_learning", "experience": 3, "location": "Los Angeles", "salary": 150000, "availability": "Immediately"},
        {"id": 8, "title": "Full Stack Developer", "skills": "javascript nodejs react sql", "experience": 4, "location": "Denver", "salary": 125000, "availability": "1-2 Weeks"},
        {"id": 9, "title": "Product Manager", "skills": "product_management agile", "experience": 6, "location": "New York", "salary": 145000, "availability": "Flexible"},
        {"id": 10, "title": "QA Engineer", "skills": "testing automation python", "experience": 2, "location": "San Jose", "salary": 105000, "availability": "Not Available"},
        {"id": 11, "title": "Business Analyst", "skills": "business_analysis sql", "experience": 3, "location": "Philadelphia", "salary": 98000, "availability": "1 Month"},
        {"id": 12, "title": "Systems Administrator", "skills": "linux windows networking", "experience": 5, "location": "Houston", "salary": 115000, "availability": "1-2 Weeks"},
        {"id": 13, "title": "UI/UX Designer", "skills": "design adobe_sketch user_research", "experience": 3, "location": "San Diego", "salary": 110000, "availability": "Immediately"},
        {"id": 14, "title": "Cloud Engineer", "skills": "aws azure gcp", "experience": 4, "location": "Atlanta", "salary": 135000, "availability": "1 Month"},
        {"id": 15, "title": "Network Engineer", "skills": "networking cisco security", "experience": 4, "location": "Dallas", "salary": 120000, "availability": "Flexible"},
        {"id": 16, "title": "Salesforce Developer", "skills": "salesforce apex visualforce", "experience": 3, "location": "San Francisco", "salary": 115000, "availability": "1-2 Weeks"},
        {"id": 17, "title": "Game Developer", "skills": "unity c# game_design", "experience": 2, "location": "Los Angeles", "salary": 125000, "availability": "Not Available"},
        {"id": 18, "title": "Digital Marketing Specialist", "skills": "seo google_ads content_marketing", "experience": 3, "location": "New York", "salary": 95000, "availability": "Immediately"},
        {"id": 19, "title": "Data Scientist", "skills": "python data_analysis machine_learning", "experience": 4, "location": "Chicago", "salary": 140000, "availability": "Flexible"},
        {"id": 20, "title": "Technical Writer", "skills": "technical_writing documentation", "experience": 5, "location": "Austin", "salary": 105000, "availability": "1 Month"},
        {"id": 21, "title": "Business Intelligence Analyst", "skills": "bi_tools sql data_visualization", "experience": 3, "location": "Philadelphia", "salary": 102000, "availability": "1-2 Weeks"},
        {"id": 22, "title": "DevOps Engineer", "skills": "docker kubernetes ci_cd", "experience": 4, "location": "Boston", "salary": 125000, "availability": "Flexible"},
        {"id": 23, "title": "Frontend Developer", "skills": "html css javascript react", "experience": 2, "location": "San Francisco", "salary": 100000, "availability": "Immediately"},
        {"id": 24, "title": "Backend Developer", "skills": "java spring boot database_management", "experience": 5, "location": "Seattle", "salary": 130000, "availability": "1-2 Weeks"},
        {"id": 25, "title": "Machine Learning Engineer", "skills": "tensorflow keras machine_learning", "experience": 3, "location": "Denver", "salary": 145000, "availability": "Flexible"},
        {"id": 26, "title": "UI/UX Designer", "skills": "figma user_research wireframing", "experience": 3, "location": "San Diego", "salary": 115000, "availability": "Not Available"},
        {"id": 27, "title": "Data Engineer", "skills": "python sql data_pipeline", "experience": 4, "location": "Atlanta", "salary": 125000, "availability": "1 Month"},
        {"id": 28, "title": "Cloud Solutions Architect", "skills": "aws azure gcp cloud_architecture", "experience": 6, "location": "Los Angeles", "salary": 150000, "availability": "Flexible"},
        {"id": 29, "title": "Network Administrator", "skills": "networking tcp_ip vpn", "experience": 4, "location": "Dallas", "salary": 110000, "availability": "Immediately"},
        {"id": 30, "title": "IT Support Specialist", "skills": "technical_support troubleshooting", "experience": 2, "location": "Chicago", "salary": 85000, "availability": "Flexible"},
        {"id": 31, "title": "Web Developer", "skills": "html css javascript", "experience": 1, "location": "New York", "salary": 90000, "availability": "Not Available"},
        {"id": 32, "title": "Full Stack Developer", "skills": "javascript nodejs react", "experience": 4, "location": "San Francisco", "salary": 130000, "availability": "1-2 Weeks"},
        {"id": 33, "title": "Software Engineer", "skills": "python java algorithms", "experience": 3, "location": "Seattle", "salary": 120000, "availability": "1 Month"},
        {"id": 34, "title": "Data Analyst", "skills": "r sql data_analysis", "experience": 2, "location": "Chicago", "salary": 95000, "availability": "Flexible"},
        {"id": 35, "title": "DevOps Engineer", "skills": "docker kubernetes ci_cd", "experience": 3, "location": "Austin", "salary": 125000, "availability": "Immediately"},
        {"id": 36, "title": "Backend Developer", "skills": "java spring boot microservices", "experience": 5, "location": "San Diego", "salary": 130000, "availability": "1-2 Weeks"},
        {"id": 37, "title": "Frontend Developer", "skills": "html css javascript", "experience": 2, "location": "Philadelphia", "salary": 105000, "availability": "Not Available"},
        {"id": 38, "title": "Machine Learning Engineer", "skills": "python machine_learning deep_learning", "experience": 3, "location": "Denver", "salary": 140000, "availability": "Flexible"},
        {"id": 39, "title": "UI/UX Designer", "skills": "adobe_sketch wireframing prototyping", "experience": 4, "location": "San Francisco", "salary": 120000, "availability": "1 Month"},
        {"id": 40, "title": "Cloud Engineer", "skills": "aws azure cloud_services", "experience": 5, "location": "Los Angeles", "salary": 135000, "availability": "1-2 Weeks"},
        {"id": 41, "title": "Product Manager", "skills": "product_management agile", "experience": 7, "location": "New York", "salary": 150000, "availability": "Flexible"},
        {"id": 42, "title": "QA Engineer", "skills": "manual_testing automation_testing", "experience": 3, "location": "San Jose", "salary": 100000, "availability": "Immediately"},
        {"id": 43, "title": "Systems Administrator", "skills": "linux windows network_security", "experience": 4, "location": "Houston", "salary": 115000, "availability": "1-2 Weeks"},
        {"id": 44, "title": "Business Analyst", "skills": "business_analysis data_analysis", "experience": 4, "location": "Philadelphia", "salary": 105000, "availability": "1 Month"},
        {"id": 45, "title": "Data Scientist", "skills": "python machine_learning data_visualization", "experience": 5, "location": "Chicago", "salary": 150000, "availability": "Flexible"},
        {"id": 46, "title": "IT Support Specialist", "skills": "customer_service technical_support", "experience": 1, "location": "Dallas", "salary": 80000, "availability": "Not Available"},
        {"id": 47, "title": "Technical Writer", "skills": "technical_documentation writing", "experience": 2, "location": "Austin", "salary": 90000, "availability": "Flexible"},
        {"id": 48, "title": "Business Intelligence Analyst", "skills": "bi_tools data_visualization", "experience": 3, "location": "New York", "salary": 100000, "availability": "1-2 Weeks"},
        {"id": 49, "title": "Cloud Solutions Architect", "skills": "aws azure cloud_computing", "experience": 6, "location": "San Diego", "salary": 150000, "availability": "Immediately"},
        {"id": 50, "title": "Network Engineer", "skills": "network_design vpn cisco", "experience": 3, "location": "Dallas", "salary": 110000, "availability": "Flexible"},
        {"id": 51, "title": "Salesforce Developer", "skills": "salesforce apex visualforce", "experience": 0, "location": "San Francisco", "salary": 115000, "availability": "1 Month"},
        {"id": 52, "title": "Game Developer", "skills": "unity c# game_design", "experience": 2, "location": "Los Angeles", "salary": 125000, "availability": "Immediately"},
        {"id": 53, "title": "Digital Marketing Specialist", "skills": "seo google_ads content_marketing", "experience": 4, "location": "New York", "salary": 100000, "availability": "Flexible"},
        {"id": 54, "title": "Data Scientist", "skills": "python data_analysis machine_learning", "experience": 5, "location": "Chicago", "salary": 145000, "availability": "1-2 Weeks"},
        {"id": 55, "title": "Technical Writer", "skills": "technical_writing documentation", "experience": 0, "location": "Austin", "salary": 95000, "availability": "Not Available"},
        {"id": 56, "title": "Business Intelligence Analyst", "skills": "bi_tools sql data_visualization", "experience": 4, "location": "Philadelphia", "salary": 107000, "availability": "Flexible"},
        {"id": 57, "title": "DevOps Engineer", "skills": "docker kubernetes ci_cd", "experience": 5, "location": "Boston", "salary": 130000, "availability": "1 Month"},
        {"id": 58, "title": "Frontend Developer", "skills": "html css javascript", "experience": 1, "location": "San Diego", "salary": 92000, "availability": "Flexible"},
        {"id": 59, "title": "Backend Developer", "skills": "java spring boot database_management", "experience": 4, "location": "Seattle", "salary": 125000, "availability": "1-2 Weeks"},
        {"id": 60, "title": "Machine Learning Engineer", "skills": "tensorflow keras deep_learning", "experience": 3, "location": "Denver", "salary": 140000, "availability": "Immediately"},
        {"id": 61, "title": "UI/UX Designer", "skills": "figma user_research prototyping", "experience": 4, "location": "San Francisco", "salary": 120000, "availability": "1 Month"},
        {"id": 62, "title": "Cloud Engineer", "skills": "aws azure cloud_services", "experience": 6, "location": "Los Angeles", "salary": 140000, "availability": "1-2 Weeks"},
        {"id": 63, "title": "Product Manager", "skills": "product_management agile", "experience": 5, "location": "New York", "salary": 150000, "availability": "Flexible"},
        {"id": 64, "title": "QA Engineer", "skills": "manual_testing automation_testing", "experience": 0, "location": "San Jose", "salary": 100000, "availability": "Immediately"},
        {"id": 65, "title": "Systems Administrator", "skills": "linux windows networking", "experience": 5, "location": "Houston", "salary": 115000, "availability": "1-2 Weeks"},
        {"id": 66, "title": "Business Analyst", "skills": "business_analysis data_visualization", "experience": 3, "location": "Philadelphia", "salary": 98000, "availability": "Flexible"},
        {"id": 67, "title": "Data Scientist", "skills": "python machine_learning data_visualization", "experience": 4, "location": "Chicago", "salary": 145000, "availability": "1 Month"},
        {"id": 68, "title": "IT Support Specialist", "skills": "technical_support troubleshooting", "experience": 0, "location": "Dallas", "salary": 80000, "availability": "Flexible"},
        {"id": 69, "title": "Technical Writer", "skills": "technical_documentation writing", "experience": 3, "location": "Austin", "salary": 100000, "availability": "1-2 Weeks"},
        {"id": 70, "title": "Business Intelligence Analyst", "skills": "bi_tools sql data_analysis", "experience": 4, "location": "New York", "salary": 110000, "availability": "Flexible"},
        {"id": 71, "title": "Cloud Solutions Architect", "skills": "aws azure gcp cloud_architecture", "experience": 6, "location": "San Diego", "salary": 150000, "availability": "Immediately"},
        {"id": 72, "title": "Network Engineer", "skills": "network_design cisco vpn", "experience": 3, "location": "Dallas", "salary": 115000, "availability": "Flexible"},
        {"id": 73, "title": "Salesforce Developer", "skills": "salesforce apex visualforce", "experience": 2, "location": "San Francisco", "salary": 105000, "availability": "Not Available"},
        {"id": 74, "title": "Game Developer", "skills": "unity c# game_design", "experience": 4, "location": "Los Angeles", "salary": 130000, "availability": "1-2 Weeks"},
        {"id": 75, "title": "Digital Marketing Specialist", "skills": "seo google_ads social_media", "experience": 3, "location": "New York", "salary": 95000, "availability": "Flexible"},
        {"id": 76, "title": "Data Scientist", "skills": "python data_analysis machine_learning", "experience": 5, "location": "Chicago", "salary": 150000, "availability": "Immediately"},
        {"id": 77, "title": "Technical Writer", "skills": "technical_writing documentation", "experience": 4, "location": "Austin", "salary": 105000, "availability": "1 Month"},
        {"id": 78, "title": "Business Intelligence Analyst", "skills": "bi_tools sql data_analysis", "experience": 3, "location": "Philadelphia", "salary": 99000, "availability": "1-2 Weeks"},
        {"id": 79, "title": "DevOps Engineer", "skills": "docker kubernetes cloud", "experience": 4, "location": "Boston", "salary": 125000, "availability": "Flexible"},
        {"id": 80, "title": "Frontend Developer", "skills": "html css javascript react", "experience": 3, "location": "San Diego", "salary": 105000, "availability": "Immediately"},
        {"id": 81, "title": "Backend Developer", "skills": "java spring boot microservices", "experience": 2, "location": "Seattle", "salary": 115000, "availability": "1-2 Weeks"},
        {"id": 82, "title": "Machine Learning Engineer", "skills": "tensorflow keras machine_learning", "experience": 3, "location": "Denver", "salary": 145000, "availability": "1 Month"},
        {"id": 83, "title": "UI/UX Designer", "skills": "figma prototyping wireframing", "experience": 4, "location": "San Francisco", "salary": 125000, "availability": "Flexible"},
        {"id": 84, "title": "Cloud Engineer", "skills": "aws azure cloud_architecture", "experience": 6, "location": "Los Angeles", "salary": 140000, "availability": "Immediately"},
        {"id": 85, "title": "Product Manager", "skills": "product_management agile", "experience": 7, "location": "New York", "salary": 160000, "availability": "Flexible"},
        {"id": 86, "title": "QA Engineer", "skills": "manual_testing automation_testing", "experience": 3, "location": "San Jose", "salary": 105000, "availability": "1-2 Weeks"},
        {"id": 87, "title": "Systems Administrator", "skills": "linux windows networking", "experience": 4, "location": "Houston", "salary": 120000, "availability": "1 Month"},
        {"id": 88, "title": "Business Analyst", "skills": "business_analysis data_analysis", "experience": 5, "location": "Philadelphia", "salary": 105000, "availability": "Flexible"},
        {"id": 89, "title": "Data Scientist", "skills": "python data_analysis machine_learning", "experience": 4, "location": "Chicago", "salary": 150000, "availability": "1 Month"},
        {"id": 90, "title": "IT Support Specialist", "skills": "technical_support troubleshooting", "experience": 0, "location": "Dallas", "salary": 85000, "availability": "Flexible"},
        {"id": 91, "title": "Technical Writer", "skills": "technical_documentation writing", "experience": 0, "location": "Austin", "salary": 95000, "availability": "1-2 Weeks"},
        {"id": 92, "title": "Business Intelligence Analyst", "skills": "bi_tools data_visualization sql", "experience": 4, "location": "New York", "salary": 108000, "availability": "1 Month"},
        {"id": 93, "title": "Cloud Solutions Architect", "skills": "aws azure cloud_computing", "experience": 6, "location": "San Diego", "salary": 155000, "availability": "Immediately"},
        {"id": 94, "title": "Network Engineer", "skills": "network_design vpn network_security", "experience": 4, "location": "Dallas", "salary": 120000, "availability": "Flexible"},
        {"id": 95, "title": "Salesforce Developer", "skills": "salesforce apex lightning", "experience": 0, "location": "San Francisco", "salary": 110000, "availability": "1-2 Weeks"},
        {"id": 96, "title": "Game Developer", "skills": "unreal_engine c++ game_design", "experience": 0, "location": "Los Angeles", "salary": 130000, "availability": "Flexible"},
        {"id": 97, "title": "Digital Marketing Specialist", "skills": "content_marketing seo google_ads", "experience": 3, "location": "New York", "salary": 95000, "availability": "1 Month"},
        {"id": 98, "title": "Data Scientist", "skills": "python machine_learning data_analysis", "experience": 5, "location": "Chicago", "salary": 155000, "availability": "Immediately"},
        {"id": 99, "title": "Technical Writer", "skills": "technical_documentation editing", "experience": 4, "location": "Austin", "salary": 102000, "availability": "Flexible"},
        {"id": 100, "title": "Business Intelligence Analyst", "skills": "bi_tools data_analysis", "experience": 4, "location": "Philadelphia", "salary": 103000, "availability": "1-2 Weeks"}
    ]



    # user_profile = {
    #     "skills": input("Enter Your Skills: "),
    #     "experience": int(input("Enter your Experience: ")),
    #     "location": input("Enter your Preferred Location: "),
    #     "salary": int(input("Enter your Expected Salary: ")),
    #     "availability": input("Enter your availability: ")
    # }

    # matched_jobs = predict_job_matches(user_profile, jobs, model, vectorizer, location_encoder, availability_encoder, scaler)

    # print("Matched Jobs:")
    # if matched_jobs:
    #     for job in matched_jobs:
    #         print(f"id: {job['job']['id']}, Job Title: {job['job']['title']}, Skills: {job['job']['skills']}, Experience: {job['job']['experience']}, Availability: {job['job']['availability']}, Match Score: {job['match_score']:.2f}")
    # else:
    #     print("No jobs matched.")

    return jsonify({"message": "Model trained successfully"})

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, vectorizer, location_encoder, availability_encoder, scaler, jobs
    
    if model is None:
        return jsonify({"error": "Model not trained. Please call /api/train first."})
    
    user_profile = request.json
    matched_jobs = predict_job_matches(user_profile, jobs, model, vectorizer, location_encoder, availability_encoder, scaler)
    
    return jsonify(matched_jobs)



if __name__ == "__main__":
    app.run(debug=True)
