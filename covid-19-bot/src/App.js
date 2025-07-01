import logo from './logo.svg';
import './App.css';
import Question from './Question.js';

function App() {
  return (
    <div className="App" >
      { <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" style={{ position: 'absolute', top: '20px', left: '50px' }} />
        <div>
          Questions 
        </div>
      </header> }
      
      <Question/> 
      
    </div>
  );
}


export default App;
