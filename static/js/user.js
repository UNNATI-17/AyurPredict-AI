// User Page Functionality
document.getElementById('symptomForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const symptoms = document.getElementById('symptoms').value;
    
    if (!symptoms.trim()) {
        alert('Please enter your symptoms');
        return;
    }
    
    // Hide form and show loading
    document.getElementById('symptomForm').style.display = 'none';
    document.getElementById('loadingState').classList.remove('d-none');
    
    // Simulate API call
    fetch('/api/analyze-symptoms', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms: symptoms })
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => {
            displayResults(data);
        }, 2000);
    })
    .catch(error => {
        console.error('Error:', error);
        // Display mock results for demonstration
        setTimeout(() => {
            displayMockResults(symptoms);
        }, 2000);
    });
});

// Add symptom from quick select
function addSymptom(symptom) {
    const textarea = document.getElementById('symptoms');
    const currentValue = textarea.value.trim();
    
    if (currentValue) {
        textarea.value = currentValue + ', ' + symptom;
    } else {
        textarea.value = symptom;
    }
}

// Display results
function displayResults(data) {
    document.getElementById('loadingState').classList.add('d-none');
    document.getElementById('resultsContainer').classList.remove('d-none');
    
    const resultsContent = document.getElementById('resultsContent');
    
    let html = '';
    
    if (data.herbs && data.herbs.length > 0) {
        data.herbs.forEach((herb, index) => {
            html += `
                <div class="herb-recommendation mb-4">
                    <div class="card" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 1.5rem;">
                        <h5 class="gradient-text">${index + 1}. ${herb.name}</h5>
                        <p class="text-dark mb-2">${herb.description}</p>
                        <div class="mb-2">
                            <strong class="text-success">Benefits:</strong>
                            <ul class="mt-2">
                                ${herb.benefits.map(b => `<li class="text-dark">${b}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="mb-2">
                            <strong class="text-warning">Precautions:</strong>
                            <p class="text-dark mt-1">${herb.precautions}</p>
                        </div>
                        <div class="confidence-badge mt-2">
                            <span class="badge bg-success">Confidence: ${herb.confidence}%</span>
                        </div>
                    </div>
                </div>
            `;
        });
    }
    
    resultsContent.innerHTML = html;
}

// Display mock results for demonstration
function displayMockResults(symptoms) {
    const mockData = {
        herbs: [
            {
                name: 'Ashwagandha',
                description: 'A powerful adaptogenic herb that helps reduce cortisol levels and supports stress management.',
                benefits: [
                    'Reduces stress and anxiety',
                    'Improves sleep quality',
                    'Enhances cognitive function',
                    'Supports immune system'
                ],
                precautions: 'Avoid combining with sedative medications. Consult doctor if pregnant or nursing.',
                confidence: 92
            },
            {
                name: 'Tulsi (Holy Basil)',
                description: 'A sacred herb with adaptogenic properties that helps balance stress hormones.',
                benefits: [
                    'Natural stress reliever',
                    'Supports respiratory health',
                    'Anti-inflammatory properties',
                    'Boosts immunity'
                ],
                precautions: 'May lower blood sugar. Monitor if diabetic.',
                confidence: 88
            },
            {
                name: 'Brahmi',
                description: 'An excellent brain tonic that improves memory and reduces mental fatigue.',
                benefits: [
                    'Enhances memory and concentration',
                    'Reduces anxiety',
                    'Supports nervous system',
                    'Improves mental clarity'
                ],
                precautions: 'May cause drowsiness. Start with small doses.',
                confidence: 85
            }
        ]
    };
    
    displayResults(mockData);
}
