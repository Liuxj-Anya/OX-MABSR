import { useState } from 'react';
import {
  Upload,
  Send,
  Image as ImageIcon,
  FileText,
  Sparkles,
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

const taskTypes = [
  {
    id: 1,
    title: 'Implicit Entity Extraction',
    description: 'Extract implicit entities and their complex sentiments from image-text pairs.',
    prompt: 'Extract Implicit entity and its Complex Sentiment'
  },
  {
    id: 2,
    title: 'Shallow Reasoning',
    description: 'Use aesthetic info, facial expressions, and scene context to judge sentiment.',
    prompt: 'Shallow reasoning using visual cues'
  },
  {
    id: 3,
    title: 'Deep Reasoning',
    description: 'Leverage historical context, events, and background information for analysis.',
    prompt: 'Deep reasoning with historical context'
  },
  {
    id: 4,
    title: 'Chain-of-Thought Analysis',
    description: 'Step-by-step multi-level reasoning from multiple angles and perspectives.',
    prompt: 'Think step by step, multi-level analysis'
  }
];

// Sample demo responses for simulation
const sampleResponses = {
  1: {
    entities: [
      { name: 'Product Quality', sentiment: 'positive', confidence: 92 },
      { name: 'Customer Service', sentiment: 'negative', confidence: 78 },
      { name: 'Delivery Time', sentiment: 'neutral', confidence: 65 }
    ],
    reasoning: 'Based on the visual content and textual description, the product quality appears to be highly regarded with positive visual cues. However, customer service aspects show negative indicators through complaint-related language. Delivery timing remains ambiguous without clear temporal markers.'
  },
  2: {
    entities: [
      { name: 'Visual Appeal', sentiment: 'positive', confidence: 88 },
      { name: 'User Experience', sentiment: 'positive', confidence: 82 },
      { name: 'Brand Perception', sentiment: 'neutral', confidence: 71 }
    ],
    reasoning: 'The aesthetic information in the image shows well-composed visuals with harmonious color schemes. Facial expressions (if present) indicate satisfaction. The scene context suggests a professional and appealing presentation that positively influences sentiment.'
  },
  3: {
    entities: [
      { name: 'Historical Context', sentiment: 'positive', confidence: 85 },
      { name: 'Cultural Significance', sentiment: 'positive', confidence: 79 },
      { name: 'Social Impact', sentiment: 'neutral', confidence: 68 }
    ],
    reasoning: 'Considering the historical background and important events associated with this content, the analysis reveals positive sentiment rooted in cultural significance. The social impact remains to be fully determined, requiring broader contextual understanding.'
  },
  4: {
    entities: [
      { name: 'Overall Sentiment', sentiment: 'positive', confidence: 90 },
      { name: 'Entity A', sentiment: 'positive', confidence: 87 },
      { name: 'Entity B', sentiment: 'negative', confidence: 73 },
      { name: 'Entity C', sentiment: 'neutral', confidence: 66 }
    ],
    reasoning: 'Step 1: Analyzing visual features — the image contains positive aesthetic elements.\nStep 2: Processing textual content — mixed sentiments detected in the description.\nStep 3: Cross-modal fusion — combining visual and textual signals reveals dominant positive sentiment.\nStep 4: Entity-level analysis — individual entities show varied sentiment patterns.\nStep 5: Final synthesis — overall positive sentiment with nuanced entity-level variations.'
  }
};

function Demo() {
  const [textInput, setTextInput] = useState('');
  const [selectedTask, setSelectedTask] = useState(4);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = () => {
    if (!textInput.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    setError('');
    setIsAnalyzing(true);
    setResult(null);

    // Simulate API call delay
    setTimeout(() => {
      setResult(sampleResponses[selectedTask]);
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleReset = () => {
    setTextInput('');
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError('');
  };

  return (
    <div className="demo-container fade-in">
      <div className="demo-header">
        <h2>Interactive Demo</h2>
        <p>
          Upload an image and provide text to perform multimodal aspect-based sentiment reasoning.
          Select a task type and click analyze to see results.
        </p>
      </div>

      <div className="demo-layout">
        {/* Input Panel */}
        <div className="demo-panel">
          <h3>
            <FileText size={20} />
            Input
          </h3>

          {/* Text Input */}
          <div className="form-group">
            <label>Text Description</label>
            <textarea
              placeholder="Enter the text description associated with your image... For example: 'The new restaurant downtown has amazing food but the service could be improved.'"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
            />
          </div>

          {/* Image Upload */}
          <div className="form-group">
            <label>Image (Optional for demo)</label>
            <label className="upload-zone">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Preview"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '200px',
                    borderRadius: '8px',
                    objectFit: 'contain'
                  }}
                />
              ) : (
                <>
                  <Upload size={32} style={{ color: 'var(--text-muted)' }} />
                  <p>Click to upload an image or drag and drop</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    PNG, JPG, GIF up to 10MB
                  </p>
                </>
              )}
            </label>
          </div>

          {/* Task Selection */}
          <div className="form-group">
            <label>Task Type</label>
            <div className="task-selector">
              {taskTypes.map((task) => (
                <div
                  key={task.id}
                  className={`task-option ${selectedTask === task.id ? 'selected' : ''}`}
                  onClick={() => setSelectedTask(task.id)}
                >
                  <h4>{task.title}</h4>
                  <p>{task.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
            <button
              className="btn btn-gradient"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              style={{ flex: 1, justifyContent: 'center' }}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={18} className="loading" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Analyze Sentiment
                </>
              )}
            </button>
            <button
              className="btn btn-outline"
              onClick={handleReset}
              style={{ justifyContent: 'center' }}
            >
              Reset
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              color: 'var(--danger)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.9rem'
            }}>
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <ImageIcon size={20} />
            Results
          </h3>

          {isAnalyzing ? (
            <div className="results-placeholder loading">
              <Loader2 size={48} style={{ color: 'var(--primary)' }} />
              <p>Analyzing sentiment with {taskTypes.find(t => t.id === selectedTask)?.title}...</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                This may take a few moments
              </p>
            </div>
          ) : result ? (
            <div>
              {/* Entity Tags */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Detected Entities & Sentiments
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {result.entities.map((entity, index) => (
                    <span key={index} className={`entity-tag ${entity.sentiment}`}>
                      {entity.sentiment === 'positive' && <CheckCircle2 size={14} />}
                      {entity.sentiment === 'negative' && <AlertCircle size={14} />}
                      {entity.name}: {entity.sentiment}
                    </span>
                  ))}
                </div>
              </div>

              {/* Confidence Bars */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Confidence Scores
                </h4>
                {result.entities.map((entity, index) => (
                  <div key={index} className="confidence-bar">
                    <label>
                      <span>{entity.name}</span>
                      <span>{entity.confidence}%</span>
                    </label>
                    <div className="bar-bg">
                      <div
                        className={`bar-fill ${entity.sentiment}`}
                        style={{ width: `${entity.confidence}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Reasoning */}
              <div>
                <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Reasoning Process
                </h4>
                <div className="reasoning-text">
                  {result.reasoning.split('\n').map((line, i) => (
                    <p key={i} style={{ marginBottom: '0.5rem' }}>{line}</p>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="results-placeholder">
              <Sparkles size={48} style={{ color: 'var(--text-muted)' }} />
              <p>Results will appear here after analysis</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Enter text and select a task type to begin
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Demo Note */}
      <div style={{
        marginTop: '2rem',
        padding: '1.25rem',
        background: 'rgba(99, 102, 241, 0.05)',
        border: '1px solid rgba(99, 102, 241, 0.2)',
        borderRadius: '12px',
        textAlign: 'center'
      }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          <strong>Note:</strong> This is a frontend demo with simulated responses. To run the actual OX-MABSR model,
          connect this interface to the Python backend API. See the{' '}
          <a href="https://github.com/Liuxj-Anya/OX-MABSR" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary-light)' }}>
            GitHub repository
          </a>{' '}
          for the full implementation.
        </p>
      </div>
    </div>
  );
}

export default Demo;
