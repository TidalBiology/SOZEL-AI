document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('inputForm');
    const inputField = document.getElementById('userInput');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', function (e) {
        e.preventDefault();  // Formun sayfayı yenilemesini engeller

        const userInput = inputField.value;

        if (!userInput) {
            resultDiv.innerHTML = "<p class='error'>Please enter some text!</p>";
            return;
        }

        // API'ye istek gönder
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_data: userInput })  // Anahtar burada 'input_data' olmalı
        })
        .then(response => response.json())
        .then(data => {
            if (data.generated_word) {
                resultDiv.innerHTML = `<p>Generated Word: ${data.generated_word}</p>`;
            } else {
                resultDiv.innerHTML = "<p class='error'>No word generated.</p>";
            }
        })
        .catch(error => {
            resultDiv.innerHTML = "<p class='error'>There was an error processing your request.</p>";
        });
    });
});
