from flask import Flask, render_template, request, redirect, url_for, session
# from pathlib import Path
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2 as cv
import pandas as pd
# import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'fingerprint'

json_file = open('models/model.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("models/model.h5")


# model._make_predict_function()


def predict_label(original_fp_img, original_iris_img, input_fp, input_iris):
        image_fp_1 = cv.imread(original_fp_img)
        image_fp_1 = cv.resize(image_fp_1, (90, 90))
        image_fp_1 = cv.cvtColor(image_fp_1, cv.COLOR_BGR2GRAY)
        image_fp_1 = np.expand_dims(image_fp_1, axis=2)
        image_fp_1 = np.expand_dims(image_fp_1, axis=0)

        image_iris_1 = cv.imread(original_iris_img)
        image_iris_1 = cv.resize(image_iris_1, (90, 90))
        image_iris_1 = cv.cvtColor(image_iris_1, cv.COLOR_BGR2GRAY)
        image_iris_1 = np.expand_dims(image_iris_1, axis=2)
        image_iris_1 = np.expand_dims(image_iris_1, axis=0)

        image_fp_2 = cv.imread(input_fp)
        image_fp_2 = cv.resize(image_fp_2, (90, 90))
        image_fp_2 = cv.cvtColor(image_fp_2, cv.COLOR_BGR2GRAY)
        image_fp_2 = np.expand_dims(image_fp_2, axis=2)
        image_fp_2 = np.expand_dims(image_fp_2, axis=0)

        image_iris_2 = cv.imread(input_iris)
        image_iris_2 = cv.resize(image_iris_2, (90, 90))
        image_iris_2 = cv.cvtColor(image_iris_2, cv.COLOR_BGR2GRAY)
        image_iris_2 = np.expand_dims(image_iris_2, axis=2)
        image_iris_2 = np.expand_dims(image_iris_2, axis=0)

        prediction1 = model.predict([image_fp_1, image_fp_2])
        score1 = prediction1
        prediction2 = model.predict([image_iris_1, image_iris_2])
        score2 = prediction2
        total_avg_score = (prediction1 + prediction2) / 2
        # msg1 = (" Total Matching Score:", total_avg_score)
        msg1 = total_avg_score
        if total_avg_score > .80:
            msg2 = "Authentication Success"
            return score1, score2, msg1, msg2
        else:
            msg3 = "Authentication Failed"
            return score1, score2, msg1, msg3


@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        org_fp = request.files['img1']
        inp_fp = request.files['img2']
        org_iris = request.files['img3']
        inp_iris = request.files['img4']
        original_fp_img = "static/" + org_fp.filename
        org_fp.save(original_fp_img)
        original_iris_img = "static/" + org_iris.filename
        org_iris.save(original_iris_img)
        input_fp = "static/" + inp_fp.filename
        inp_fp.save(input_fp)
        input_iris = "static/" + inp_iris.filename
        inp_iris.save(input_iris)

        p = predict_label(original_fp_img, original_iris_img, input_fp, input_iris)
        print(p)
        return render_template("result.html", prediction=p, original_fp_img=original_fp_img,
                               original_iris_img=original_iris_img, input_fp=input_fp, input_iris=input_iris)


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):

                return redirect(url_for('home'))
        else:
            msg = 'Invalid Login Try Again'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('register.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


@app.route('/vgg')
def vgg():
    return render_template('lstm.html')


@app.route('/cnn')
def cnn():
    return render_template('cnn.html')


@app.route('/logout')
def logout():
    session.clear()
    msg = 'You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(port=3000, debug=True)
