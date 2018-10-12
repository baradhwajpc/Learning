# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:32:27 2018

@author: baradhwaj
"""

'''

Task 1 : Set up the Environment.
Once the virtual server is provided, click on `Run --> Install` to install required dependencies and perform the below specified Task 2.

Task 2 : Create a Basic Flask Application

After setting up the environment, Open the file 'helloapp/hello.py' and follow the instructions specified in it.
Create a flask application named 'app' in it.
Define a view function named 'hello', which returns the string "Hello World!!! I've run my first Flask application."
Ensure that the view function 'hello' is routed when a user access the URL '/' .
Run the application at 0.0.0.0 and port 8000.

Task 3: Viewing the Application
After completing creating the application, You can access the application by clicking on `Run ---> Run Server`
This enables a 'PreviewApp' button. You can click on it to open the home page of the flask application in hackerrank specific browser.
In case if you are not able to view the application in hackerrank specific browser, copy and paste the home page url in your local browser.
This opens home page of the flask application.

Task 4: Testing the Application

After completing the application, you can test it by clicking on 'Run --> Test'.
It displays how many test cases have passed out of total test cases defined in `helloapp/tests.py`.
In case if any test case fails, modify the contents of 'helloapp/hello.py' and ensure that all test cases pass.
 

Task 4: Submit the Solution

Once all test cases are passed, you can click on 'Submit' to submit your solution to hackerrank.
'''
from flask import Flask

## Define a flask application name 'app' below
app = Flask(__name__)
@app.route('/')
## Define below a view function 'hello', which displays the message
## "Hello World!!! I've run my first Flask application."
def hello():
    return "Hello World!!! I've run my first Flask application."

## Write the required code below which runs flask applictaion 'app' defined above
## on host 0.0.0.0 and port 8000
if __name__ == '__main__':
    app.run(host="0.0.0.0",port="8000")