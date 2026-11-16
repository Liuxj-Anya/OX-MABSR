import {
  BookOpen,
  Users,
  Database,
  Cpu,
  GitBranch,
  Download,
  ExternalLink
} from 'lucide-react';

function About() {
  return (
    <div className="about-container fade-in">
      <div className="about-header">
        <h2>About OX-MABSR</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
          Understanding the research behind multimodal sentiment reasoning
        </p>
      </div>

      {/* Paper Info */}
      <div className="paper-card">
        <h3>
          OX-MABSR: A Benchmark for Open-domain Explainable Multimodal Aspect-Based Sentiment Reasoning
        </h3>
        <div className="paper-meta">
          <span>AAAI 2026</span>
          <span>Natural Language Processing</span>
          <span>Multimodal AI</span>
          <span>Sentiment Analysis</span>
        </div>

        <div className="authors-list">
          <span className="author-chip">Liu X.J.</span>
          <span className="author-chip">Anya</span>
        </div>

        <p>
          OX-MABSR introduces a comprehensive benchmark for open-domain explainable multimodal
          aspect-based sentiment reasoning. Unlike traditional sentiment analysis that focuses on
          single-modal inputs, our approach combines visual and textual information to perform
          deep, explainable sentiment reasoning across multiple levels of analysis.
        </p>

        <p>
          The benchmark addresses the challenge of understanding complex sentiments in real-world
          scenarios where both images and text contribute to the overall emotional context. Our
          model leverages the Qwen3-VL architecture with LoRA fine-tuning to achieve state-of-the-art
          performance on multimodal sentiment reasoning tasks.
        </p>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
          <a
            href="https://github.com/Liuxj-Anya/OX-MABSR"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-gradient"
            style={{ fontSize: '0.9rem', padding: '0.6rem 1.25rem' }}
          >
            <GitBranch size={16} />
            GitHub Repository
          </a>
          <a
            href="https://drive.google.com/drive/folders/1ayN8fIQF1aLegnNIVM8pFiegRI8BQKMo?usp=sharing"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-outline"
            style={{ fontSize: '0.9rem', padding: '0.6rem 1.25rem' }}
          >
            <Download size={16} />
            Download Dataset
          </a>
        </div>
      </div>

      {/* Methodology */}
      <div className="methodology">
        <h3>
          <BookOpen size={24} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Methodology
        </h3>
        <div className="method-steps">
          <div className="method-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Multimodal Input Processing</h4>
              <p>
                The system accepts both image and text inputs, processing them through separate
                modalities before fusion. Images are encoded using visual feature extractors,
                while text is tokenized using the Qwen3 tokenizer.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Image Feature Extraction</h4>
              <p>
                Visual features are extracted from input images using pretrained vision encoders.
                These features capture aesthetic information, facial expressions, scene context,
                and other visual cues relevant to sentiment analysis.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Cross-Modal Fusion</h4>
              <p>
                The image projector fuses visual features with textual embeddings, enabling the
                model to reason about sentiment using both modalities simultaneously. Special
                tokens are used to mark image positions in the sequence.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Multi-Level Reasoning</h4>
              <p>
                The model performs reasoning at multiple levels: shallow reasoning using immediate
                visual cues, deep reasoning incorporating historical context, and chain-of-thought
                analysis for comprehensive multi-angle sentiment understanding.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">5</div>
            <div className="step-content">
              <h4>Entity & Sentiment Extraction</h4>
              <p>
                The final output includes identified entities, their associated sentiments
                (positive, negative, neutral), confidence scores, and detailed reasoning
                explanations for each sentiment judgment.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture */}
      <div className="methodology">
        <h3>
          <Cpu size={24} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Model Architecture
        </h3>
        <div className="method-steps">
          <div className="method-step">
            <div className="step-number">
              <Database size={18} />
            </div>
            <div className="step-content">
              <h4>Base Model: Qwen3-14B</h4>
              <p>
                Built on the Qwen3-14B language model, providing strong foundational
                language understanding and generation capabilities.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">
              <Cpu size={18} />
            </div>
            <div className="step-content">
              <h4>LoRA Fine-tuning</h4>
              <p>
                Parameter-efficient fine-tuning using LoRA adapters, allowing the model to
                specialize in multimodal sentiment reasoning without full model retraining.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">
              <Users size={18} />
            </div>
            <div className="step-content">
              <h4>Training Strategies</h4>
              <p>
                Multiple training approaches including standard supervised fine-tuning,
                Chain-of-Thought training, and GRPO-based optimization for improved reasoning.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Metrics */}
      <div className="methodology">
        <h3>
          <BookOpen size={24} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Evaluation Metrics
        </h3>
        <div className="method-steps">
          <div className="method-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>BLEU Score</h4>
              <p>
                Measures the quality of generated text by comparing n-gram overlap with
                reference outputs.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>ROUGE Score</h4>
              <p>
                Evaluates summary-level similarity between generated and reference responses
                using recall-oriented metrics.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>LLM-based Evaluation</h4>
              <p>
                Uses large language models as judges to assess the quality, coherence, and
                accuracy of sentiment reasoning outputs.
              </p>
            </div>
          </div>
          <div className="method-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Aspect-level Accuracy</h4>
              <p>
                Measures the accuracy of entity identification and sentiment classification
                at the aspect level.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Citation */}
      <div className="paper-card" style={{ marginTop: '2rem' }}>
        <h3>Citation</h3>
        <p>If you use this benchmark or model in your research, please cite:</p>
        <div style={{
          background: 'var(--bg-dark)',
          border: '1px solid var(--border)',
          borderRadius: '10px',
          padding: '1.25rem',
          fontFamily: 'monospace',
          fontSize: '0.85rem',
          color: 'var(--text-secondary)',
          marginTop: '1rem',
          whiteSpace: 'pre-wrap'
        }}>
{`@inproceedings{liu2026oxmabsr,
  title={OX-MABSR: A Benchmark for Open-domain Explainable Multimodal Aspect-Based Sentiment Reasoning},
  author={Liu, X.J. and Anya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}`}
        </div>
      </div>
    </div>
  );
}

export default About;
