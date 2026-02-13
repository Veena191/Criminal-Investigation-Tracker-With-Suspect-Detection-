from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crime.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODELS ----------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))
    role = db.Column(db.String(20))


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crime_type = db.Column(db.String(50))
    location = db.Column(db.String(50))
    time_of_day = db.Column(db.String(50))
    status = db.Column(db.String(30))

    evidences = db.relationship('Evidence', backref='case', lazy=True, cascade="all, delete")


class Suspect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    criminal_history = db.Column(db.String(200))


class Evidence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey('case.id'), nullable=False)
    evidence_type = db.Column(db.String(50))
    description = db.Column(db.String(200))


# ---------------- ML TRAINING ----------------

def train_model():
    data = pd.read_csv('training_data.csv')

    le_crime = LabelEncoder()
    le_location = LabelEncoder()
    le_time = LabelEncoder()

    data['crime_type'] = le_crime.fit_transform(data['crime_type'])
    data['location'] = le_location.fit_transform(data['location'])
    data['time_of_day'] = le_time.fit_transform(data['time_of_day'])

    X = data[['crime_type', 'location', 'time_of_day']]
    y = data['suspect']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump((model, le_crime, le_location, le_time), 'ml_model.pkl')


# ---------------- HOME ----------------

@app.route('/')
def home():
    cases = Case.query.all()
    return render_template('index.html', cases=cases)


# ---------------- USER MANAGEMENT ----------------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    user = User(
        username=data['username'],
        password=data['password'],
        role=data['role']
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"})


# ---------------- CASE MANAGEMENT ----------------

@app.route('/add_case', methods=['POST'])
def add_case():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"})

    case = Case(
        crime_type=data.get('crime_type'),
        location=data.get('location'),
        time_of_day=data.get('time_of_day'),
        status="Open"
    )

    db.session.add(case)
    db.session.commit()

    return jsonify({"message": "Case added successfully"})


@app.route('/view_cases', methods=['GET'])
def view_cases():
    cases = Case.query.all()
    result = []

    for c in cases:
        evidence_list = []
        for e in c.evidences:
            evidence_list.append({
                "id": e.id,
                "evidence_type": e.evidence_type,
                "description": e.description
            })

        result.append({
            "id": c.id,
            "crime_type": c.crime_type,
            "location": c.location,
            "time_of_day": c.time_of_day,
            "status": c.status,
            "evidence": evidence_list
        })

    return jsonify(result)


@app.route('/delete_case/<int:case_id>', methods=['DELETE'])
def delete_case(case_id):
    case = Case.query.get(case_id)

    if not case:
        return jsonify({"error": "Case not found"})

    db.session.delete(case)
    db.session.commit()

    return jsonify({"message": "Case deleted successfully"})


# ---------------- SUSPECT MANAGEMENT ----------------

@app.route('/add_suspect', methods=['POST'])
def add_suspect():
    data = request.get_json()

    suspect = Suspect(
        name=data['name'],
        criminal_history=data['criminal_history']
    )

    db.session.add(suspect)
    db.session.commit()

    return jsonify({"message": "Suspect added successfully"})


@app.route('/delete_suspect/<int:suspect_id>', methods=['DELETE'])
def delete_suspect(suspect_id):
    suspect = Suspect.query.get(suspect_id)

    if not suspect:
        return jsonify({"error": "Suspect not found"})

    db.session.delete(suspect)
    db.session.commit()

    return jsonify({"message": "Suspect deleted successfully"})


# ---------------- EVIDENCE MANAGEMENT ----------------

@app.route('/add_evidence', methods=['POST'])
def add_evidence():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    case_id = data.get('case_id')
    evidence_type = data.get('evidence_type')
    description = data.get('description')

    if not case_id:
        return jsonify({"error": "case_id is required"}), 400

    case = Case.query.get(case_id)

    if not case:
        return jsonify({"error": "Case not found"}), 404

    new_evidence = Evidence(
        case_id=case_id,
        evidence_type=evidence_type,
        description=description
    )

    db.session.add(new_evidence)
    db.session.commit()

    return jsonify({"message": "Evidence added successfully"})

@app.route('/view_evidence/<int:case_id>', methods=['GET'])
def view_evidence(case_id):
    evidences = Evidence.query.filter_by(case_id=case_id).all()

    result = []
    for e in evidences:
        result.append({
            "id": e.id,
            "case_id": e.case_id,
            "evidence_type": e.evidence_type,
            "description": e.description
        })

    return jsonify(result)


@app.route('/delete_evidence/<int:evidence_id>', methods=['DELETE'])
def delete_evidence(evidence_id):
    evidence = Evidence.query.get(evidence_id)

    if not evidence:
        return jsonify({"error": "Evidence not found"})

    db.session.delete(evidence)
    db.session.commit()

    return jsonify({"message": "Evidence deleted successfully"})


# ---------------- ML SUSPECT PREDICTION ----------------

@app.route('/predict_suspect', methods=['POST'])
def predict_suspect():
    data = request.get_json()

    if not os.path.exists('ml_model.pkl'):
        return jsonify({"error": "ML model not trained"})

    model, le_crime, le_location, le_time = joblib.load('ml_model.pkl')

    try:
        input_data = [[
            le_crime.transform([data['crime_type']])[0],
            le_location.transform([data['location']])[0],
            le_time.transform([data['time_of_day']])[0]
        ]]
    except:
        return jsonify({"error": "Invalid input values"})

    prediction = model.predict(input_data)

    return jsonify({"suspect_likely": str(prediction[0])})


# ---------------- REPORT ----------------

@app.route('/report', methods=['GET'])
def report():
    return jsonify({
        "total_cases": Case.query.count(),
        "open_cases": Case.query.filter_by(status="Open").count(),
        "total_suspects": Suspect.query.count(),
        "total_evidence": Evidence.query.count()
    })


# ---------------- MAIN ----------------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        if not os.path.exists('ml_model.pkl'):
            train_model()

    app.run(debug=True)