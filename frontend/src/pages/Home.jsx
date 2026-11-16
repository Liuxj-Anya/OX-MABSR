import { Link } from 'react-router-dom';
import {
  Brain,
  Image,
  MessageSquare,
  TrendingUp,
  Zap,
  Globe,
  Award,
  ChevronRight
} from 'lucide-react';

function Home() {
  const features = [
    {
      icon: <Brain size={24} />,
      iconClass: 'blue',
      title: 'Multimodal Reasoning',
      description: 'Combines visual and textual information to perform deep aspect-based sentiment analysis with explainable reasoning.'
    },
    {
      icon: <Image size={24} />,
      iconClass: 'green',
      title: 'Image-Text Fusion',
      description: 'Processes both images and text inputs simultaneously using advanced Qwen3-VL architecture with LoRA fine-tuning.'
    },
    {
      icon: <MessageSquare size={24} />,
      iconClass: 'orange',
      title: 'Entity Extraction',
      description: 'Identifies entities in multimodal inputs and determines their complex sentiments with detailed explanations.'
    },
    {
      icon: <TrendingUp size={24} />,
      iconClass: 'red',
      title: 'Multi-Level Analysis',
      description: 'Performs shallow reasoning using aesthetics and expressions, and deep reasoning using historical context and background.'
    },
    {
      icon: <Zap size={24} />,
      iconClass: 'cyan',
      title: 'Chain-of-Thought',
      description: 'Step-by-step reasoning process that obtains sentiments and causes from multiple levels and angles.'
    },
    {
      icon: <Globe size={24} />,
      iconClass: 'purple',
      title: 'Open-Domain Benchmark',
      description: 'A comprehensive benchmark dataset for open-domain explainable multimodal aspect-based sentiment reasoning.'
    }
  ];

  const stats = [
    { number: '6', label: 'Task Types' },
    { number: '50K+', label: 'Data Samples' },
    { number: 'Qwen3', label: 'Base Model' },
    { number: 'AAAI', label: '2026 Paper' }
  ];

  return (
    <div className="fade-in">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-badge">
          <Award size={16} />
          AAAI 2026 Publication
        </div>
        <h2>
          Open-domain Explainable<br />
          <span className="highlight">Multimodal Aspect-Based</span><br />
          Sentiment Reasoning
        </h2>
        <p>
          A state-of-the-art benchmark and model for understanding complex sentiments
          in multimodal content through explainable, multi-level reasoning.
        </p>
        <div className="hero-buttons">
          <Link to="/demo" className="btn btn-gradient">
            Try Demo <ChevronRight size={18} />
          </Link>
          <Link to="/about" className="btn btn-outline">
            Read Paper
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="section-title">
          <h3>Key Capabilities</h3>
          <p>Advanced multimodal sentiment analysis with explainable reasoning</p>
        </div>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className={`feature-icon ${feature.iconClass}`}>
                {feature.icon}
              </div>
              <h4>{feature.title}</h4>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
        <div className="section-title">
          <h3>Benchmark at a Glance</h3>
          <p>Comprehensive dataset and evaluation framework</p>
        </div>
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <div key={index} className="stat-card">
              <div className="stat-number">{stat.number}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Task Types Section */}
      <section className="features">
        <div className="section-title">
          <h3>Supported Tasks</h3>
          <p>Multiple reasoning levels for comprehensive sentiment analysis</p>
        </div>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon blue">1</div>
            <h4>Implicit Entity Extraction</h4>
            <p>Identify hidden entities and their complex sentiments from multimodal inputs.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon green">2</div>
            <h4>Shallow Reasoning</h4>
            <p>Use aesthetic information, facial expressions, and scene context to judge sentiment.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon orange">3</div>
            <h4>Deep Reasoning</h4>
            <p>Leverage historical context, important events, and background information for analysis.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon cyan">4</div>
            <h4>Chain-of-Thought</h4>
            <p>Step-by-step multi-level reasoning to obtain sentiments and causes from multiple angles.</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="hero" style={{ paddingBottom: '6rem' }}>
        <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Ready to Explore?</h3>
        <p style={{ marginBottom: '2rem' }}>
          Try our interactive demo to see OX-MABSR in action with your own image-text pairs.
        </p>
        <Link to="/demo" className="btn btn-gradient">
          Start Analyzing <ChevronRight size={18} />
        </Link>
      </section>
    </div>
  );
}

export default Home;
