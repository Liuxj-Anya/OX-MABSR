# OX-MABSR Frontend

A modern, interactive web interface for the **OX-MABSR** (Open-domain Explainable Multimodal Aspect-Based Sentiment Reasoning) benchmark and model.

## 🚀 Features

- **Landing Page**: Overview of the OX-MABSR benchmark, key capabilities, and supported tasks
- **Interactive Demo**: Upload images and text to perform multimodal sentiment analysis with simulated results
- **About Page**: Detailed information about the paper, methodology, model architecture, and evaluation metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark Theme**: Modern dark UI with gradient accents

## 🛠️ Tech Stack

- **React 19** - UI library
- **Vite** - Build tool and development server
- **React Router DOM** - Client-side routing
- **Lucide React** - Icon library

## 📦 Installation

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx       # Navigation bar component
│   │   └── Footer.jsx       # Footer component
│   ├── pages/
│   │   ├── Home.jsx         # Landing page
│   │   ├── Demo.jsx         # Interactive demo page
│   │   └── About.jsx        # About paper page
│   ├── App.jsx              # Main app with routing
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── index.html               # HTML template
├── package.json             # Dependencies
└── vite.config.js           # Vite configuration
```

## 🎯 Pages

### Home (`/`)
- Hero section with AAAI 2026 badge
- Key capabilities showcase
- Benchmark statistics
- Supported task types
- Call-to-action for demo

### Demo (`/demo`)
- Text input for descriptions
- Image upload with preview
- Task type selection (4 types)
- Real-time analysis simulation
- Entity detection with sentiment tags
- Confidence score visualization
- Reasoning process display

### About (`/about`)
- Paper information and metadata
- Author details
- Methodology explanation
- Model architecture overview
- Evaluation metrics
- BibTeX citation

## 🔌 Backend Integration

This frontend currently uses simulated responses for demonstration. To connect with the actual OX-MABSR Python backend:

1. Set up the Python backend API (see `CODE/eval_server.py`)
2. Replace the `setTimeout` simulation in `Demo.jsx` with actual API calls
3. Configure CORS settings on the backend
4. Update the API endpoint URLs

Example API integration:
```javascript
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: textInput,
    task_id: selectedTask,
    image: imageFile
  })
});
const data = await response.json();
setResult(data);
```

## 📄 License

This project is part of the OX-MABSR benchmark. See the main repository for licensing information.

## 📚 Citation

```bibtex
@inproceedings{liu2026oxmabsr,
  title={OX-MABSR: A Benchmark for Open-domain Explainable Multimodal Aspect-Based Sentiment Reasoning},
  author={Liu, X.J. and Anya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 🔗 Links

- [GitHub Repository](https://github.com/Liuxj-Anya/OX-MABSR)
- [Download Dataset](https://drive.google.com/drive/folders/1ayN8fIQF1aLegnNIVM8pFiegRI8BQKMo?usp=sharing)
