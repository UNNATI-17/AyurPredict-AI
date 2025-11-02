from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/researcher')
def researcher():
    return render_template('researcher.html')

@app.route('/api/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    
    # TODO: Integrate your ML model here
    # For now, returning mock data
    
    response = {
        'herbs': [
            {
                'name': 'Ashwagandha',
                'description': 'A powerful adaptogenic herb',
                'benefits': ['Reduces stress', 'Improves sleep'],
                'precautions': 'Avoid with sedatives',
                'confidence': 92
            }
        ]
    }
    
    return jsonify(response)

@app.route('/api/analyze-herb', methods=['POST'])
def analyze_herb():
    data = request.get_json()
    herb = data.get('herb', '')
    
    # TODO: Integrate your ML model here
    # For now, returning mock data
    
    response = {
        'compounds': ['Withanolides', 'Alkaloids'],
        'targets': ['NF-κB', 'AMPK'],
        'pathways': ['Inflammation pathway'],
        'effects': ['Anti-inflammatory'],
        'safety': 'Generally safe',
        'confidence': 90
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
