function Quesiton() {
    const options = ["yes", "no"];
    return(
    <div>
        <div>
            Have you had Tuberculosis?
        </div>

        <div>
            <select>
            
            {options.map((option) => (
            <option key={option} value={option}>
                {option}
            </option>
            ))}
            </select>
        </div>

        <div>
            <button>
                next
            </button>
        </div>
    </div>

    );

}

export default Quesiton;