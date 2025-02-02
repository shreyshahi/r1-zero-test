import { useState, useEffect } from 'react';
import './App.css';
import React from 'react';

function App() {
  const [currentData, setCurrentData] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const loadRandomFile = async () => {
      try {
        // Fetch the manifest file that contains the list of all files
        const manifestResponse = await fetch('/data/manifest.json');
        const fileList = await manifestResponse.json();
        
        // Pick a random file from the list
        const randomFile = fileList[Math.floor(Math.random() * fileList.length)];
        
        // Fetch the content of the random file
        const dataResponse = await fetch(`/data/${randomFile}`);
        const data = await dataResponse.json();
        
        setCurrentData(data);
        // Set initial step based on available steps
        const steps = Object.keys(data.steps).sort((a, b) => Number(a) - Number(b));
        setCurrentStep(steps[0]);
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadRandomFile();
  }, []);

  if (!currentData) {
    return <div>Loading...</div>;
  }

  const handleSliderChange = (e) => {
    const steps = Object.keys(currentData.steps).sort((a, b) => Number(a) - Number(b));
    const stepIndex = Math.min(Math.max(0, parseInt(e.target.value)), steps.length - 1);
    setCurrentStep(steps[stepIndex]);
  };

  const steps = Object.keys(currentData.steps).sort((a, b) => Number(a) - Number(b));
  const currentStepIndex = steps.indexOf(currentStep);

  const formatChatMessage = (text) => {
    if (!text) return null;
    
    return text.split('<|eot_id|>').map((segment, index) => {
      if (segment.includes('<|start_header_id|>')) {
        const role = segment.split('<|start_header_id|>')[1].split('<|end_header_id|>')[0];
        const content = segment.split('<|end_header_id|>')[1];
        return (
          <div key={index} className={`message ${role}`}>
            <span className="role">{role}:</span>
            <span className="content">{content.split('\n').map((line, i) => (
              <React.Fragment key={i}>
                {line}
                {i < content.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}</span>
          </div>
        );
      }
      return null;
    });
  };

  return (
    <div className="App">
      <h1>Question Viewer</h1>
      <div className="two-column-layout">
        <div className="column question-column">
          <div className="question-box">
            <h2>Question:</h2>
            <div className="chat-messages">
              {formatChatMessage(currentData.question)}
            </div>
          </div>
        </div>

        <div className="column answer-column">
          <button onClick={() => window.location.reload()} className="refresh-button">
            Load Another Question
          </button>

          <div className="slider-container">
            <input
              type="range"
              min={0}
              max={steps.length - 1}
              value={currentStepIndex}
              onChange={handleSliderChange}
              className="slider"
            />
            <div className="step-label">Step: {currentStep} tokens</div>
          </div>

          <div className="answer-box">
            <h2>Response for Step {currentStep}:</h2>
            <div className="chat-messages">
              {formatChatMessage(currentData.steps[currentStep]?.response)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
