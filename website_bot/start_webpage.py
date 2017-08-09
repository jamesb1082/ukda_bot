from flask import Flask, request, render_template, redirect, url_for 
from chatbot import Chatbot

app = Flask(__name__) 

@app.route('/')
def redir(): 
    return redirect(url_for('display_answer')) 

@app.route('/chatbot', methods=['POST', 'GET']) 
def display_answer():
    print(type(c) ) 
    question1=""
    if request.method == 'POST':
        question1 = request.form['question'].strip() 
        return render_template("webpage.html", answer=chatbot.get_answer(question1), q=question1)
    else:
        print("default page loaded") 
        return render_template("webpage.html", answer="", question="") 

if __name__ == '__main__':
    c = Chatbot()
    app.debug = True
    app.run() 
    app.run(debug= True) 
