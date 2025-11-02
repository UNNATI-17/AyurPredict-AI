// Landing Page Animations
document.addEventListener('DOMContentLoaded', function() {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all highlight cards
    document.querySelectorAll('.highlight-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.6s ease';
        observer.observe(card);
    });
});

// Show Mode Selection
function showModeSelection() {
    const landingPage = document.getElementById('landingPage');
    const modeSelection = document.getElementById('modeSelection');
    
    landingPage.style.opacity = '0';
    setTimeout(() => {
        landingPage.classList.add('d-none');
        modeSelection.classList.remove('d-none');
        setTimeout(() => {
            modeSelection.style.opacity = '1';
        }, 50);
    }, 300);
}

// Redirect to appropriate mode
function redirectToMode(mode) {
    if (mode === 'user') {
        window.location.href = '/user';
    } else if (mode === 'researcher') {
        window.location.href = '/researcher';
    }
}

// Add smooth transitions
document.querySelectorAll('.mode-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});
