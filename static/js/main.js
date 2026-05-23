// main.js — WeatherAI Flask App

// ── Navbar scroll effect ──────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('mainNav');
  if (nav) {
    nav.style.background = window.scrollY > 50
      ? 'rgba(6,11,20,0.98)'
      : 'rgba(6,11,20,0.85)';
  }
});

// ── Intersection observer for fade-in cards ───────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity = '1';
        e.target.style.transform = 'translateY(0)';
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.feature-card, .plot-card, .glass-card, .mini-stat-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });
});

// ── Predict form loading state ────────────────────────────────────────────────
const predForm = document.getElementById('predForm');
if (predForm) {
  predForm.addEventListener('submit', function () {
    const btn = document.getElementById('predictBtn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting…';
    }
  });
}
