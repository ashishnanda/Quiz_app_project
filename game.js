const question = document.getElementById('question');
const choices = Array.from(document.getElementsByClassName("choice-text"));
console.log(choices);


let currentQuestion = {};
let acceptingAnswers = false;
let score = 0;
let questionCounter = 0;
let availableQuesions = [];

let questions = [
    {
        question: 'Inside which HTML element do we put the JavaScript??',
        choice1: '<script>',
        choice2: '<javascript>',
        choice3: '<js>',
        choice4: '<scripting>',
        answer: 1,
    },
    {
        question:
            "What is the correct syntax for referring to an external script called 'xxx.js'?",
        choice1: "<script href='xxx.js'>",
        choice2: "<script name='xxx.js'>",
        choice3: "<script src='xxx.js'>",
        choice4: "<script file='xxx.js'>",
        answer: 3,
    },
    {
        question: " How do you write 'Hello World' in an alert box?",
        choice1: "msgBox('Hello World');",
        choice2: "alertBox('Hello World');",
        choice3: "msg('Hello World');",
        choice4: "alert('Hello World');",
        answer: 4,
    },
];

/* CONSTANTS */
const CORRECT_MARKS = 3;
const INCORRECT_MARKS = -1
const MAX_QUESTIONS = 3;

startGame = () => {
    questionCounter = 0;
    score = 0;
    availableQuesions = [...questions]
    getNewQuestion();
};

getNewQuestion = () => {
    
    if(availableQuesions.length == 0 || questionCounter >= MAX_QUESTIONS){
        return window.location.assign('./end.html')
    }
    questionCounter++;
    const questionIndex = Math.floor(Math.random() * availableQuesions.length);
    currentQuestion = availableQuesions[questionIndex];
    question.innerText = currentQuestion.question;

    choices.forEach(choice => {
        const num  = choice.dataset['number'];
        choice.innerText = currentQuestion['choice' + num]
    });

    availableQuesions.splice(questionIndex, 1);
    //console.log(availableQuesions)
    acceptingAnswers = true
};


choices.forEach(choice => {
    choice.addEventListener('click', e => {
        if(!acceptingAnswers)  return;

        acceptingAnswers = false;

        const selectedChoice = e.target;
        const selectedAnswer = selectedChoice.dataset['number'];
        
        //console.log(selectedAnswer == currentQuestion.answer);
        
        const classToApply = 
        selectedAnswer == currentQuestion.answer ? 'correct' : 'incorrect';
        
        // APPLY  COLOR TO DENOTE CORRECT OR INCORRECT TO THE SELECTED OPTION
        selectedChoice.parentElement.classList.add(classToApply);
        //console.log(classToApply)

        //  ADD DELAY
        setTimeout(()  => {

            // REMOVE THE COLOR CODING WHICH IS STUCK TO THE OPTION NUMBER
            selectedChoice.parentElement.classList.remove(classToApply);
            getNewQuestion();
        }, 1000);
        
    });
});


startGame();