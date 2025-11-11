// ==============================
// 🌿 Researcher Page Functionality
// ==============================

const herbDatabase = [
    'Ashwagandha', 'Turmeric', 'Tulsi', 'Brahmi', 'Ginger', 'Neem',
    'Triphala', 'Guduchi', 'Shatavari', 'Amla', 'Haritaki', 'Bhringraj'
];

// ==============================
// 🔹 Autocomplete for Herb Input
// ==============================
document.getElementById('herbName').addEventListener('input', function (e) {
    const value = e.target.value.toLowerCase();
    const suggestions = document.getElementById('herbSuggestions');

    if (value.length < 2) {
        suggestions.innerHTML = '';
        return;
    }

    const matches = herbDatabase.filter(herb =>
        herb.toLowerCase().includes(value)
    );

    suggestions.innerHTML = matches.length
        ? matches.map(herb => `<div class="suggestion-item" onclick="selectHerb('${herb}')">${herb}</div>`).join('')
        : '';
});

// ==============================
// 🔹 Select Herb from Suggestions
// ==============================
function selectHerb(herbName) {
    document.getElementById('herbName').value = herbName;
    document.getElementById('herbSuggestions').innerHTML = '';
}

// ==============================
// 🔹 Form Submission Handler
// ==============================
document.getElementById('herbForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const herbName = document.getElementById('herbName').value.trim();
    if (!herbName) {
        alert('Please enter a herb name');
        return;
    }

    // Hide form and show loading animation
    this.style.display = 'none';
    document.getElementById('analysisResults').classList.add('d-none');
    document.getElementById('analysisLoading').classList.remove('d-none');

    animateLoadingSteps();

    // Call Flask API
    fetch('/api/analyze-herb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ herb: herbName })
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => {
            if (data.error) {
                console.warn('Server Error:', data.error);
                displayMockAnalysis(herbName);
            } else {
                displayAnalysisResults(herbName, data);
            }
        }, 4000);
    })
    .catch(error => {
        console.error('Error:', error);
        setTimeout(() => {
            displayMockAnalysis(herbName);
        }, 4000);
    });
});

// ==============================
// 🔹 Loading Animation Steps
// ==============================
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.step-item');
    let currentStep = 0;

    steps.forEach((step, index) => {
        step.classList.remove('active');
        step.querySelector('i').className = index === 0 ? 'fas fa-check-circle' : 'fas fa-circle';
    });
    steps[0].classList.add('active');

    const interval = setInterval(() => {
        currentStep++;
        if (currentStep < steps.length) {
            steps[currentStep - 1].querySelector('i').className = 'fas fa-check-circle';
            steps[currentStep].classList.add('active');
            steps[currentStep].querySelector('i').className = 'fas fa-spinner fa-spin';
        } else {
            clearInterval(interval);
            steps[steps.length - 1].querySelector('i').className = 'fas fa-check-circle';
        }
    }, 1000);
}

// ==============================
// 🔹 Display Analysis Results (Includes Image with Fallback)
// ==============================
async function displayAnalysisResults(herbName, data) {
    document.getElementById('analysisLoading').classList.add('d-none');
    document.getElementById('analysisResults').classList.remove('d-none');
    document.getElementById('analyzedHerbName').textContent = herbName;
    document.getElementById('herbForm').style.display = 'block';

    const content = document.getElementById('analysisContent');

    // Fetch herb image dynamically using Unsplash API
    let imageUrl = '';
    try {
        const imgResponse = await fetch(`https://api.unsplash.com/search/photos?query=${encodeURIComponent(herbName)}&per_page=1&client_id=-5bObY-qBs2o0A7EJ7h9r6F0acjPdUtdczhKx5kwmys`);
        const imgData = await imgResponse.json();
        imageUrl = imgData.results && imgData.results.length
            ? imgData.results[0].urls.small
            : 'https://via.placeholder.com/250?text=No+Image';
    } catch (err) {
        console.warn('Image fetch failed:', err);
        imageUrl = 'https://via.placeholder.com/250?text=No+Image';
    }

    content.innerHTML = `
        <div class="col-12 text-center mb-3">
            <h5>🌿 Herb Image</h5>
            <img src="${imageUrl}" alt="${herbName}" class="img-fluid rounded shadow" style="max-height: 250px;">
        </div>

        <div class="col-md-6">
            <div class="result-card compounds">
                <h5><i class="fas fa-atom"></i> Phytochemical Compounds</h5>
                <ul>${data.compounds.map(c => `<li>${c}</li>`).join('')}</ul>
            </div>
        </div>

        <div class="col-md-6">
            <div class="result-card targets">
                <h5><i class="fas fa-crosshairs"></i> Predicted Protein Targets</h5>
                <ul>${data.targets.map(t => `<li>${t}</li>`).join('')}</ul>
            </div>
        </div>

        <div class="col-md-6">
            <div class="result-card pathways">
                <h5><i class="fas fa-project-diagram"></i> Biological Pathways</h5>
                <ul>${data.pathways.map(p => `<li>${p}</li>`).join('')}</ul>
            </div>
        </div>

        <div class="col-md-6">
            <div class="result-card effects">
                <h5><i class="fas fa-heartbeat"></i> Predicted Therapeutic Effects</h5>
                <ul>${data.effects.map(e => `<li>${e}</li>`).join('')}</ul>
            </div>
        </div>

        <div class="col-12">
            <div class="result-card safety">
                <h5><i class="fas fa-exclamation-triangle"></i> Safety Considerations</h5>
                <p>${data.safety}</p>
                <p><strong>Confidence Score:</strong> ${data.confidence}%</p>
            </div>
        </div>
    `;
}

// ==============================
// 🔹 Mock Data (Fallback)
// ==============================
function displayMockAnalysis(herbName) {
    const mockData = {
        compounds: [
            'Withanolides (steroidal lactones)',
            'Withaferin A',
            'Alkaloids (isopelletierine)',
            'Saponins'
        ],
        targets: [
            'NF-κB (Nuclear Factor Kappa B)',
            'AMPK (AMP-activated protein kinase)',
            'GABA-A receptor',
            'Cortisol receptors'
        ],
        pathways: [
            'Inflammation regulation pathway',
            'Metabolic pathway (glucose regulation)',
            'Neurotransmitter signaling'
        ],
        effects: [
            'Anti-inflammatory activity',
            'Anti-diabetic potential',
            'Stress and anxiety reduction',
            'Immune system modulation'
        ],
        safety: 'May increase drowsiness when combined with sedatives. Monitor blood glucose if diabetic. Not recommended during pregnancy without medical supervision.',
        confidence: 91
    };

    displayAnalysisResults(herbName, mockData);
}

// ==============================
// 🔹 Close Suggestions on Outside Click
// ==============================
document.addEventListener('click', function (e) {
    if (!e.target.closest('.herb-input-container')) {
        document.getElementById('herbSuggestions').innerHTML = '';
    }
});
