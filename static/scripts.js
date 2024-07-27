function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            };

            reader.readAsDataURL(input.files[0]);
        }
    }

    function uploadImage() {
        const formData = new FormData(uploadForm);
        fetch('/', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.Issue) {
                predictionResult.textContent = `Issue: ${data.Issue}\nExplanation: ${data["Explanation of the issue"]}\nWhat to do: ${data["What to do"]}`;
            } else {
                predictionResult.textContent = 'Error: Could not analyze the image. Please try again.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            predictionResult.textContent = 'An error occurred while trying to analyze the image.';
        });
    }

    imageUpload.addEventListener('change', function() {
        readURL(this);
    });
