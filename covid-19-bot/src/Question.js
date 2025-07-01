import axios from "axios";
import { useState } from "react";
import './Question.css';

function Question() {
    const data = [{'Q':'Have you had tuberculosis!', 'O':["yes","no"]}, {"Q":"Are you pregnant?", "O":["yes","no"]}];
    const [QuestionIndex, setQuestionIndex] = useState(0);
    const [value, setValue] = useState('');
    const [answers, setAnswers] = useState([]);
    console.log(answers);
    const [num, addNum] = useState(0);
    const {diagnosis, setDiagnosis} = useState(0);

    function handleChange(event){
        setValue(event.target.value); 
      };

    function handleSubmit(){
        const data = { 'answers': answers };
        axios.post('http://localhost:8080/predict', data)
        .then(response => setDiagnosis(response.data));

    }

    
    return(
     <div className = "question-container">
         <div className = "big-quesiton" style={{ fontSize: '100px' }} >  
             {data[QuestionIndex]['Q']}
         </div>

         <div>
             <select className="big-select" value = {value} onChange ={handleChange}>
            
             {data[QuestionIndex]["O"].map((option) => (
            
             <option key={option} value={option}>
                 {option}
             </option>
                ))}

             </select>
             <p> {value}</p>
         </div>
        
        
        

         <div>
             <button onClick={() => {setQuestionIndex(QuestionIndex+1) 
                setAnswers([...answers, value])
                addNum(num + 1)}} className = "big-button">
                 next
             </button>
         </div>
         {num === 1 &&<div> 
            <button onClick = {() => handleSubmit()}>
                submit
            </button>
         </div>}

         <div>
            if (diagnosis === 0){

            }else{

                
            }
         </div>
     </div>

    );

}

export default Question;