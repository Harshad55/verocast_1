

let Signupbtn = document.querySelector(".Signupbtn");
let Signinpbtn = document.querySelector(".signinbtn");
let nameField = document.querySelector('.namefield');
let title = document.querySelector('.title');
let underline = document.querySelector('.underline');
let demo = document.querySelector('.demo');

let isSignIn = true; // Initially, consider it's sign-in

// Function to handle sign-in action
function signInAction() {
    nameField.style.maxHeight='0';
    title.innerHTML='Sign In';
    Signupbtn.classList.add('disable');
    Signinpbtn.classList.remove('disable');
    underline.style.transform= 'translateX(35px)';
}

// Function to handle redirection to home page
function redirectToHome() {
    fetch('/home')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            // You can handle the response if needed
            // For example, if you expect JSON response:
            // return response.json();
            // If you don't need to handle the response, you can just continue
            window.location.href = "/home";
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
}
// Function to toggle between sign-in action and redirection
function toggleAction() {
    if (isSignIn) {
        signInAction();
    } else {
        redirectToHome();
    }
    isSignIn = !isSignIn; // Toggle the flag
   
}


// Add event listener for the double event button
Signinpbtn.addEventListener("click", toggleAction);



Signupbtn.addEventListener('click',function(){
    nameField.style.maxHeight='60px';
    title.innerHTML='Sign Up';
    Signupbtn.classList.remove('disable');
    Signinpbtn.classList.add('disable');
    underline.style.transform= 'translateX(0)';
})








