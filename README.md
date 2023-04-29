### Image_Caption_Generator
This is a Image Caption generator Project using Google Vision Transformer (ViT) pretrained model

### Prerequisites
You must have PIL, transformers and Flask (for API) installed.

### Project Structure
This project has three major parts :.
1. app.py - This contains Flask APIs that receive image, computes 3 captions based on our model and returns it in other page.
2.input.html: Platform for user to insert image and  return it to app.py
4. output.html: Displays captions generated from Model

### Running the project
1.Make User you have Prerequisites Libraries installed else install them
```
2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://127.0.0.1:5000

You should be able to view the homepage as below :

Insert a valid Picture and hit generate captions.

If everything goes well, you should  be able to see the predicted captions

### Note:Can Use Images in test_cases folder for Verifying 
