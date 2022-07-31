from flask import Flask, render_template, request, redirect, url_for, Response, flash,session
from flask_login import login_user, login_required, logout_user, current_user
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import PySimpleGUI as sg
import re
#from werkzeug.security import generate_password_hash, check_password_hash
import pyotp
 
app = Flask(__name__)

app.secret_key = 'Webweb123@'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="toor",
    database="webimperators"
)
mycursor = mydb.cursor()
 
secret = None 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 50
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #save image as jpg
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = (r'C:\Users\Bek Zi Ying\Desktop\webimperators\webimperators\dataset')
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    
    flash("Face registered", "success")
    return redirect('/login')
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        coords = []
        #confidence = []
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            draw_boundary.confidence = int(100 * (1 - pred / 300))
            
            mycursor.execute("select b.prs_name "
                             "  from img_dataset a "
                             "  left join prs_mstr b on a.img_person = b.prs_nbr "
                             " where img_id = " + str(id))
            s = mycursor.fetchone()
            s = '' + ''.join(s)
            
            if draw_boundary.confidence > 75 :
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            #print(confidence)
            
            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)

        return img 
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    wCam, hCam = 500, 400
    #sampleNum = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
   
    
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        #print(img)
        sampleNum +=1
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        #compare facial recognition with database
        if draw_boundary.confidence > 75 :
            #access granted if facial recognition detect 75% with database
            layout = [[sg.Text("Access Granted")], [sg.Button("OK")]]

            #Create the window
            window = sg.Window("Access", layout)

            #Create an event loop
            while True:
                event, values = window.read()
                # End program if user closes window or
                # presses the OK button
                if event == "OK" or event == sg.WIN_CLOSED:
                    break
            window.close()
            cap.release()
           
        else:
             #access denied if facial recognition does not detect 75% with database
            layout = [[sg.Text("Access Denied")], [sg.Button("OK")]]

            # Create the window
            window = sg.Window("Access", layout)

            # Create an event loop
            while True:
                event, values = window.read()
                # End program if user closes window or
                # presses the OK button
                if event == "OK" or event == sg.WIN_CLOSED:
                    break

            window.close()
            cap.release()
            
            
            
        key = cv2.waitKey(1)
        if key == 27 or sampleNum > 10:       
            break
        break
    cap.release()
    cv2.destroyAllWindows()
            
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Routes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/')
def home():
    # mycursor.execute("select prs_nbr, prs_name, prs_active, prs_added from prs_mstr")
    # data = mycursor.fetchall()
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('welcome.html', username=session['username'])
    # User is not loggedin redirect to login page
    # return redirect(url_for('login'))
    return render_template('index.html')

#route for email and password authentication
@app.route('/sign_up',methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        #reset secret token for different user
        secret = pyotp.random_base32()
        
        mycursor.execute('SELECT * FROM user WHERE email = %s', (email, ))
        user = mycursor.fetchone()

        #validate registration form
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(username) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 10:
            flash('Password must be at least 10 characters.', category='error')
        elif re.search('[0-9]',password1) is None:
            flash("Make sure your password has a number in it")
        elif re.search('[A-Z]',password1) is None: 
            flash("Make sure your password has a capital letter in it")
        elif re.search('[^a-zA-Z0-9]',password1) is None: 
            flash("Make sure your password has a special character in it")
        #if all above correct, add new user to db
        else:
            # password1 = generate_password_hash(password1)
            mycursor.execute('INSERT INTO user VALUES (NULL, %s, %s, %s,%s)', (email, password1, username,secret))
            # password1 = generate_password_hash(password1)
            
            mydb.commit()
            flash('Account created!', category='success')
            # redirect to signup2fa page
            return redirect(url_for('signup_2fa'))

    return render_template("sign_up.html", user=current_user)

#route for adding facial recognition
@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
   
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`) VALUES
                    ('{}', '{}')""".format(prsnbr, prsname))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))

#route for login page with password authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # password = generate_password_hash(password, method='sha256')
        # check email and password with database
        mycursor.execute('SELECT * FROM user WHERE email = %s AND password = %s', (email, password))
        user = mycursor.fetchone()
        # for row in user:
        #     hashed_password = ("%s" % (row["password"]))
        #     userid = ("%s" % (row["user_id"]))
        if user:
            # check_password_hash(user[2],password):
            session['loggedin'] = True
            session['userid'] = user[0]
            # session['email'] = user[1]
            session['username'] = user[3]

            flash('Logged in successfully!', category='success')
            # login_user(user, remember=True) 
            return redirect(url_for('login_2fa'))
            # else:
            #     flash('Incorrect password, try again.', category='error')
        else:
            # flash('Email does not exist.', category='error')
            flash('Incorrect password or email, try again.', category='error')

    return render_template("login.html")

@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)

#route to video stream at facial recognition registration 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')

#route to video stream at facial recognition login 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    return render_template('fr_page.html')

# 2FA registration route
@app.route('/signup2fa')
def signup_2fa():
    # generating random secret key for authentication
    global secret

    if secret is None:
        secret = pyotp.random_base32()
    return render_template("signup_2fa.html", secret=secret)

#route for google authenticator registration
@app.route('/signup2fa', methods=["POST"])
def signup_2fa_form():
    # getting secret key used by user
    secret = request.form.get("secret")
    # getting OTP provided by user
    otp = int(request.form.get("otp"))

    # verifying submitted OTP with PyOTP
    if pyotp.TOTP(secret).verify(otp):
        # inform users if OTP is valid
        flash("The TOTP 2FA token is valid", "success")
        ####### redirect to facial recognition page
        return redirect(url_for('addprsn'))
    else:
        # inform users if OTP is invalid
        flash("You have supplied an invalid 2FA token!", "error")
        return redirect(url_for('signup_2fa'))

# 2FA page route
@app.route('/login2fa')
def login_2fa():
    return render_template("login_2fa.html", secret=secret)

#route for google authenticator login
@app.route('/login2fa', methods=["POST"])
def login_2fa_form():
    # getting secret key used by user
    secret = request.form.get("secret")
    # getting OTP provided by user
    otp = int(request.form.get("otp"))

    # verifying submitted OTP with PyOTP
    if pyotp.TOTP(secret).verify(otp):
        # inform users if OTP is valid
        flash("The TOTP 2FA token is valid", "success")
        # redirect to facial recognition page is success
        return redirect(url_for("fr_page"))
    else:
        # inform users if OTP is invalid
        # stay at the same login_2fa page
        flash("You have supplied an invalid 2FA token!", "error")
        return redirect(url_for("login_2fa"))

#route for success login
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

#route for failed login
@app.route('/loginFailed')
def loginFailed():
    return render_template('loginFailed.html')

#route for logout
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    flash('You are now logged out','success')
    
    return redirect(url_for('login'))

if __name__ == "__main__":
    #run web browser with host and port
    app.run(host='127.0.0.1', port=5000, debug=True)
