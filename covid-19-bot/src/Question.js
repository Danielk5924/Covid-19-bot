import axios from "axios";
import { useState } from "react";
import './Question.css';

function Question() {
    const data = [{'Q':'What is your sex?', 'O':[1, 2]}, 
    {"Q":"are you dead?", "O":[0,1]},
    {'Q':'are you connected to a ventilator?', 'O':[1,2]},
    {'Q':'Do you have pneumonia?', 'O':[1,2]},
    {'Q':'How old are you?', 'O':[50, 60, 70, 80, 90]},
    {'Q':'are you pregnant?', 'O':[1,2]},
    {'Q':'Do you have diabetes?', 'O':[1,2]},
    {'Q':'Do you have Copd', 'O':[1,2]},
    {'Q':'Do you have asthma?', 'O':[1,2]},
    {'Q':'Are you immuno suppressed?', 'O':[1,2]},
    {'Q':'do you have hypertension?', 'O':[1,2]},
    {'Q':'Any other diseases?', 'O':[1,2]},
    {'Q':'Do you have a any heard or blood realted diseases?', 'O':[1,2]},
    {'Q':'Are you obese?', 'O':[1, 2]},
    {'Q':'Do you have chronic renal disease', 'O':[1,2]},
    {'Q':'do you use tobacco?', 'O':[1,2]},
    {'Q':'Have you been admitted to ICU?', 'O':[1,2]}];

    const [QuestionIndex, setQuestionIndex] = useState(0);
    const [value, setValue] = useState('');
    const [answers, setAnswers] = useState([]);
    const [num, addNum] = useState(0);
    const [diagnosis, setDiagnosis] = useState(0);

    function handleChange(event){
        setAnswers([...answers, value]);
        addNum(num + 1);
        setQuestionIndex(QuestionIndex + 1);
        setValue(event.target.value); 
      };

    function handleSubmit(){         
        addNum(num+1);                                                                                                            
        const data = [...answers, value];
        data.forEach((val, index) => {data[index] = parseFloat(val)});
        console.log(data);
        axios.post('http://localhost:8000/predict', data, { headers: {'Content-Type': 'application/json'}})
        .then(response => setDiagnosis(response.data));
        setValue('');
    }

    
    return(
    <div>
    {num < 17 && <div className = "question-container">
         <div className = "big-quesiton" style={{ fontSize: '100px' }} >  
             {data[QuestionIndex]['Q']}
         </div>

         <div>
             <select className="big-select" value = {value} onChange ={(event) => setValue(event.target.value)}>
            <option value="" disabled hidden>
                Select an option...
            </option>
             {data[QuestionIndex]["O"].map((option) => (
             <option key={option} value={option}>
                 {option}
             </option>
                ))} 

             </select>
             <p> {value}</p>
         </div>
        
        
        

         {num !== 16 && <div>
             <button onClick={handleChange} 
                className = "big-button"
                disabled={value === ""}>
                 next
             </button>
         </div>}
         {num === 16 &&<div> 
            <button onClick = {handleSubmit}
                className = "big-button"
                disabled={value === ""}>
                submit
            </button>
         </div> 
         }
         </div>}
         
        <div>
            diagnosis: {diagnosis}
         </div>

         </div>

    );

}

export default Question;