from flask import Flask, render_template, request, redirect, url_for, session, send_file
import string
from argon2 import PasswordHasher
from dbuild import Database
import random

from captcha_creater import generate_captcha

ph = PasswordHasher()
db = Database('localhost', 'TestServer', 'password123', 'captchaorc')
app = Flask(__name__)
app.secret_key = ''.join(random.choices((string.ascii_letters+string.digits), k=128))

@app.route('/')
def index():  # put application's code here
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return render_template('index.html', user='游客')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        src = refresh_captcha()
        return render_template('register.html', src=src)


    # POST
    username = request.form['username'] # 与表单name属性一致
    password = request.form['password']
    confirm = request.form['confirm']
    captcha = request.form['captcha']

    true_code = session.get('captcha_code', '')
    if captcha.lower() != true_code.lower():
        src = refresh_captcha()
        return render_template('register.html', error="验证码错误", src=src)
    if confirm != password:
        src = refresh_captcha()
        return render_template('register.html', error="密码不一致！", src=src)

    user_check = db.execute_sql("SELECT * FROM users where username=%s", (username,))
    if user_check:
        src = refresh_captcha()
        return render_template('register.html', error="用户名已存在！", src=src)

    password = ph.hash(password)
    db.execute_sql(
        "INSERT INTO users(username, passwd) VALUES (%s, %s)",
        (username, password)
    )
    src = refresh_captcha()
    return render_template('register.html', success="注册成功！", src=src)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        src = refresh_captcha()
        return render_template("login.html", src=src)

    username = request.form['username']
    password = request.form['password']
    captcha = request.form['captcha']

    true_pass = db.execute_sql("SELECT passwd FROM users WHERE username = %s", (username,))
    true_captcha = session.get('captcha_code', '')
    if captcha.lower() != true_captcha.lower():
        src = refresh_captcha()
        return render_template('login.html', error="验证码错误", src=src)
    if not true_pass:
        src = refresh_captcha()
        return render_template("login.html", error="用户名或密码错误", src=src)
    try:
        ph.verify(true_pass[0][0], password)
        # 跳转到index()函数上
        session['user'] = username
        return redirect(url_for('index'))
    except Exception:
        src = refresh_captcha()
        return render_template('login.html', error='用户名或密码错误', src=src)

@app.route('/logout')
def logout():
    # None：当字典中没有'user'键时不抛出异常而是返回None
    session.pop('user', None)
    return redirect(url_for("index"))

@app.route('/refresh_captcha')
def refresh_captcha():
    name, code = generate_captcha('./static/CAPTCHA')
    session['captcha_code'] = code
    session['captcha_name'] = name
    # send_file将静态文件发送给浏览器
    return f'CAPTCHA/{name}.png'

if __name__ == '__main__':
    app.run()
