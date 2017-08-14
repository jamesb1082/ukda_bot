import sys
sys.path.insert(0,"../") 
from flask import Flask, request, render_template, redirect, url_for 
from snn_chatbot.chatbot import Chatbot

app = Flask(__name__) 

@app.route('/')
def redir(): 
    return redirect(url_for('display_answer')) 

@app.route('/chatbot', methods=['POST', 'GET']) 
def display_answer():
    question1=""
    if request.method == 'POST':
        question1 = unicode(request.form['question'].strip() )
        answer1 =c.get_answer(question1)
        return render_template("webpage.html", answer=answer1, q=question1)
    else:
        print("default page loaded") 
        return render_template("webpage.html", answer="", question="") 

if __name__ == '__main__':
    c = Chatbot()
    #app.debug = True
    #app.run() 
    app.run(debug= True) 
