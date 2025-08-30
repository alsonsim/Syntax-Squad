import React, { useState } from 'react';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [textareaFocused, setTextareaFocused] = useState(false);

  const mainGreen = '#00FF66';
  const darkGreen = '#003313';
  const buttonGreen = '#00FF66';
  const shadow = '0 8px 16px rgba(0,0,0,0.08)';
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review }),
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "There was an error connecting to the server." });
    }
    setLoading(false);
  };

  const styles = {
    container: {
      maxWidth: 600,
      margin: '40px auto',
      fontFamily: "'Segoe UI', 'Arial', sans-serif",
      padding: 24,
      backgroundColor: '#fff',
      borderRadius: 14,
      boxShadow: shadow,
      border: `3px solid ${mainGreen}`,
    },
    header: {
      textAlign: 'center',
      marginBottom: 32,
      color: darkGreen,
      fontSize: 36,
      fontWeight: 800,
      letterSpacing: '-1px',
    },
    form: {
      background: '#fff',
      padding: 24,
      borderRadius: 12,
      boxShadow: shadow,
      display: 'flex',
      flexDirection: 'column',
      gap: 18,
    },
    textarea: {
      width: '100%',
      padding: 16,
      fontSize: 18,
      borderRadius: 8,
      border: `2px solid ${textareaFocused ? mainGreen : '#e0e0e0'}`,
      resize: 'vertical',
      fontFamily: 'inherit',
      outline: 'none',
      boxSizing: 'border-box',
      transition: 'border-color 0.3s',
    },
    button: {
      width: '100%',
      background: buttonGreen,
      color: '#111',
      padding: 16,
      border: 'none',
      borderRadius: 8,
      fontSize: 20,
      fontWeight: '700',
      cursor: loading || !review ? 'default' : 'pointer',
      boxShadow: '0 2px 8px rgba(0,255,102,0.08)',
      transition: 'background-color 0.3s',
      marginTop: 12,
    },
    buttonDisabled: {
      background: '#b2ffd4',
      color: '#333',
    },
    result: {
      marginTop: 38,
      padding: 24,
      borderRadius: 12,
      backgroundColor: mainGreen,
      color: darkGreen,
      boxShadow: shadow,
      fontSize: 18,
      minHeight: 74,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
    },
    error: {
      color: '#d9534f',
      fontWeight: 'bold',
      background: '#fff',
      padding: 10,
      borderRadius: 8,
      border: `2px solid ${mainGreen}`,
    },
    label: {
      fontWeight: '700',
      fontSize: 18,
      margin: '12px 0 6px 0',
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>Where to Review?</div>
      <form style={styles.form} onSubmit={handleSubmit}>
        <textarea
          value={review}
          onChange={e => setReview(e.target.value)}
          rows={5}
          placeholder="Type your review here..."
          style={styles.textarea}
          onFocus={() => setTextareaFocused(true)}
          onBlur={() => setTextareaFocused(false)}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !review}
          style={
            loading || !review
              ? { ...styles.button, ...styles.buttonDisabled }
              : styles.button
          }
        >
          {loading ? 'Classifying...' : 'Classify'}
        </button>
      </form>
      {result && (
        <div style={styles.result}>
          {result.error ? (
            <span style={styles.error}>{result.error}</span>
          ) : (
            <>
              <span style={styles.label}>Category:</span> {result.label}<br />
              <span style={styles.label}>Category ID:</span> {result.label_id}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
