import { useState, useEffect } from 'react';
import './App.css';
import React from 'react';

const BASE_URL = 'https://d15tjawjl9dif0.cloudfront.net';

function App() {
  const [currentData, setCurrentData] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadRandomFile = async () => {
      try {
        // Log the URL we're trying to fetch
        console.log(`Fetching manifest from: ${BASE_URL}/data/manifest.json`);
        
        const manifestResponse = await fetch(`${BASE_URL}/data/manifest.json`);
        if (!manifestResponse.ok) {
          throw new Error(`Manifest fetch failed with status: ${manifestResponse.status}`);
        }
        const fileList = await manifestResponse.json();
        
        // Pick a random file from the list
        const randomFile = fileList[Math.floor(Math.random() * fileList.length)];
        console.log(`Fetching data from: ${BASE_URL}/data/${randomFile}`);
        
        const dataResponse = await fetch(`${BASE_URL}/data/${randomFile}`);
        if (!dataResponse.ok) {
          throw new Error(`Data fetch failed with status: ${dataResponse.status}`);
        }
        const data = await dataResponse.json();
        
        setCurrentData(data);
        // Set initial step based on available steps
        const steps = Object.keys(data.steps).sort((a, b) => Number(a) - Number(b));
        setCurrentStep(steps[0]);
      } catch (error) {
        console.error('Error loading data:', error);
        setError(error.message);
      }
    };

    loadRandomFile();
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

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
        
        // Skip system messages
        if (role.toLowerCase() === 'system') return null;
        
        // Replace 'user' with 'question' and 'assistant' with 'answer'
        const displayRole = role.toLowerCase() === 'user' ? 'question' : 'answer';
        const displayContent = role.toLowerCase() === 'user' 
          ? content.replace('Use the format described in the system prompt to answer the following question: ', '')
          : content;
        
        return (
          <div key={index} className={`message ${role}`}>
            <span className="role">{displayRole}:</span>
            <br />
            <span className="content">{displayContent.split('\n').map((line, i) => (
              <React.Fragment key={i}>
                {line}
                {i < displayContent.split('\n').length - 1 && <br />}
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
      <h1>RL Test Set Progress Viewer</h1>
      
      <button onClick={() => window.location.reload()} className="refresh-button">
        Load Another Question
      </button>

      <div className="chat-container">
        <div className="chat-messages">
          {formatChatMessage(currentData.question)}
        </div>
      </div>

      <div className="slider-container">
        <input
          type="range"
          min={0}
          max={steps.length - 1}
          value={currentStepIndex}
          onChange={handleSliderChange}
          className="slider"
        />
        <div className="step-label">Step: {currentStep} out of 1500</div>
      </div>

      <div className="response-container">
        {currentData.steps[currentStep]?.response && (
          <div className="step-response">
            {formatChatMessage(currentData.steps[currentStep].response)}
          </div>
        )}
        <div className="correct-answer">
          <h3>Correct Answer: {currentData.answer}</h3>
        </div>
      </div>

      <div className="info-section">
        <h2>About This Viewer</h2>
        <p>
          This viewer demonstrates the learning progress of a fine-tuned Llama 3.2 1B model trained using GRPO 
          (Group relative policy optimization) on the GSM8K dataset - a collection of 
          grade school math problems.
        </p>
        <p>
          Each example shown here comes from the test set, which was not used during training. The slider allows 
          you to see how the model's responses evolved throughout the training process (from step 0 to 1500). 
          The illustrative problems selected here are the ones that the model got wrong in the early training steps. 
          You'll notice that in early training steps (near 0), the model's answers are mostly incorrect, but as 
          training progresses (towards 1500), the model learns to solve these problems correctly.
        </p>
        <p>
          Use the slider to move between training steps and observe how the model's problem-solving ability 
          improves over time. Click "Load Another Question" to see a different example from the test set.
        </p>
      </div>
    </div>
  );
}

export default App;
